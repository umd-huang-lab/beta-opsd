#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vLLM-based evaluation for math datasets with math_verify scoring.

Builds a mixed prompt pool across all datasets, then reports metrics per dataset.
Supports full checkpoints and LoRA adapter checkpoints.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoints with vLLM and math_verify.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--base_model", type=str, default="", help="Base model when --checkpoint is a LoRA adapter.")
    p.add_argument("--lora_adapter", type=str, default="", help="Explicit LoRA adapter path. Defaults to --checkpoint when --base_model is set.")
    p.add_argument("--tokenizer", type=str, default="", help="Tokenizer path. Defaults to adapter/checkpoint.")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--datasets", type=str, default="", help="Comma-separated list of dataset names to evaluate (without .jsonl). Empty = all.")
    p.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--thinking", dest="thinking", action="store_true", default=True,
                   help="Enable Qwen3 thinking mode (default: on).")
    p.add_argument("--no-thinking", dest="thinking", action="store_false",
                   help="Disable Qwen3 thinking mode.")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--fewshot_seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=0, help="Max samples per dataset (0 = all).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=0,
                   help="Maximum new tokens. Use 0 or a negative value for no explicit generation cap.")
    p.add_argument("--dataset_max_tokens", type=str, default="",
                   help="Per-dataset token overrides, e.g. 'aime24:32768,aime25:32768'. Falls back to --max_new_tokens.")
    p.add_argument("--decoding", type=str, choices=["greedy", "sampling"], default="greedy")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=0,
                   help="Maximum context length. If unset or <=0, defaults to 40960 in thinking mode and 32768 otherwise.")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--disable_shuffle", action="store_true")
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--output_jsonl", type=str, default="")
    return p.parse_args()


def load_math_verify():
    try:
        from math_verify import parse, verify
        return parse, verify
    except Exception as exc:
        raise ImportError("math_verify is required. Install with: pip install math-verify") from exc


def math_verify_score(parse_fn, verify_fn, completion: str, ground_truth: Optional[str]) -> float:
    if not ground_truth:
        return 0.0
    try:
        pred_parsed = parse_fn(completion)
        gt_parsed = parse_fn(ground_truth)
        if pred_parsed is None or gt_parsed is None:
            return 0.0
        return float(verify_fn(pred_parsed, gt_parsed))
    except Exception:
        return 0.0


def build_messages(question: str, system_prompt: str, fewshot: List[Tuple[str, str]], thinking: bool) -> List[Dict[str, str]]:
    instruction = system_prompt.strip() if system_prompt.strip() else DEFAULT_SYSTEM_PROMPT
    user_message = f"{question}\n\n{instruction}"
    messages: List[Dict[str, str]] = []
    for fs_q, fs_a in fewshot:
        fs_user_message = f"{fs_q}\n\n{instruction}"
        messages.append({"role": "user", "content": fs_user_message})
        messages.append({"role": "assistant", "content": fs_a})
    messages.append({"role": "user", "content": user_message})
    return messages


def render_chat_prompt(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    thinking: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        fallback_messages = [dict(m) for m in messages]
        if thinking and fallback_messages and fallback_messages[-1]["role"] == "user":
            fallback_messages[-1]["content"] = fallback_messages[-1]["content"] + "\n/think"
        return tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_question_and_answer(row: Dict[str, Any]) -> Tuple[str, str]:
    question = row.get("question", "") if isinstance(row.get("question"), str) else ""
    answer_raw = row.get("ground_truth")
    answer = "" if answer_raw is None else str(answer_raw).strip()
    return question, answer


def load_dataset_rows(path: Path, limit: int) -> List[Dict[str, Any]]:
    ds = load_dataset("json", data_files={"test": str(path)}, split="test")
    rows = [ds[i] for i in range(len(ds))]
    if limit > 0:
        rows = rows[:limit]
    return rows


def parse_dataset_max_tokens(spec: str, default: int) -> Dict[str, int]:
    """Parse 'aime24:32768,aime25:32768' into a dict; missing keys fall back to default."""
    result: Dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        ds, _, val = part.partition(":")
        result[ds.strip()] = int(val.strip())
    return result


def build_sampling_params(args: argparse.Namespace, max_tokens: Optional[int] = None) -> SamplingParams:
    n_tokens = max_tokens if max_tokens is not None else args.max_new_tokens
    effective_max_tokens = n_tokens if n_tokens is not None and n_tokens > 0 else None
    if args.decoding == "greedy":
        return SamplingParams(n=args.k, temperature=0.0, top_p=1.0, max_tokens=effective_max_tokens)
    return SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=effective_max_tokens,
    )


def resolve_model_paths(args: argparse.Namespace) -> Tuple[str, Optional[str], str]:
    use_lora = bool(args.base_model) or bool(args.lora_adapter)
    if use_lora:
        model_path = args.base_model or args.checkpoint
        adapter_path = args.lora_adapter or args.checkpoint
        tokenizer_path = args.tokenizer or adapter_path
        return model_path, adapter_path, tokenizer_path
    model_path = args.checkpoint
    tokenizer_path = args.tokenizer or model_path
    return model_path, None, tokenizer_path


def infer_max_lora_rank(adapter_path: str) -> int:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return 64
    try:
        config = json.loads(config_path.read_text())
        return int(config.get("r", 64))
    except Exception:
        return 64


def make_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    checkpoint_path = Path(args.checkpoint.rstrip("/"))
    parts = [p for p in checkpoint_path.parts if p not in ("/", "")]
    stem = "__".join(parts[-2:]) if len(parts) >= 2 else checkpoint_path.name
    out_json = Path(args.output_json) if args.output_json else Path("eval_results") / f"{stem}.json"
    out_jsonl = Path(args.output_jsonl) if args.output_jsonl else Path("eval_results") / f"{stem}.samples.jsonl"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    return out_json, out_jsonl


def main() -> None:
    args = parse_args()
    parse_fn, verify_fn = load_math_verify()
    if args.max_model_len <= 0:
        args.max_model_len = 40960 if args.thinking else 32768

    model_path, adapter_path, tokenizer_path = resolve_model_paths(args)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs: Dict[str, Any] = dict(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if adapter_path is not None:
        llm_kwargs.update(
            enable_lora=True,
            max_lora_rank=infer_max_lora_rank(adapter_path),
            max_loras=1,
        )

    llm = LLM(**llm_kwargs)
    lora_request = None
    if adapter_path is not None:
        lora_request = LoRARequest(lora_name="eval_lora", lora_int_id=1, lora_path=adapter_path)

    ds_max_tokens = parse_dataset_max_tokens(args.dataset_max_tokens, args.max_new_tokens)

    data_dir = Path(args.data_dir)
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if args.datasets:
        wanted = set(n.strip() + ".jsonl" for n in args.datasets.split(","))
        jsonl_files = [f for f in jsonl_files if f.name in wanted]

    effective_limit = args.samples
    repeats = max(1, args.repeat)
    out_json, out_jsonl = make_output_paths(args)

    final_results: Dict[str, Any] = {"checkpoint": args.checkpoint, "runs": []}

    with out_jsonl.open("w", encoding="utf-8") as sample_writer:
        for run_idx in range(repeats):
            rng = random.Random(args.fewshot_seed + run_idx)
            mixed_examples: List[Dict[str, Any]] = []

            for jsonl_path in jsonl_files:
                rows = load_dataset_rows(jsonl_path, effective_limit)
                if not rows:
                    continue

                fewshot_pool: List[Tuple[str, str]] = []
                if args.num_fewshot > 0:
                    for row in rows:
                        fs_q, fs_a = get_question_and_answer(row)
                        if fs_q and fs_a:
                            fewshot_pool.append((fs_q, fs_a))

                for idx, row in enumerate(rows):
                    question, answer = get_question_and_answer(row)
                    if not question:
                        continue
                    fewshot: List[Tuple[str, str]] = []
                    if args.num_fewshot > 0 and fewshot_pool:
                        fewshot = rng.sample(fewshot_pool, k=min(args.num_fewshot, len(fewshot_pool)))
                    prompt = render_chat_prompt(
                        tokenizer,
                        build_messages(question, args.system_prompt, fewshot, args.thinking),
                        args.thinking,
                    )
                    mixed_examples.append({
                        "dataset": jsonl_path.stem,
                        "index": idx,
                        "question": question,
                        "answer": answer,
                        "prompt": prompt,
                        "max_new_tokens": ds_max_tokens.get(jsonl_path.stem, args.max_new_tokens),
                    })

            if not args.disable_shuffle:
                rng.shuffle(mixed_examples)

            per_dataset: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {"correct": 0, "correct_samples": 0, "total": 0}
            )

            for start in tqdm(range(0, len(mixed_examples), args.batch_size), desc=f"run {run_idx + 1}/{repeats}"):
                batch = mixed_examples[start : start + args.batch_size]
                batch_params = [build_sampling_params(args, b["max_new_tokens"]) for b in batch]
                outputs = llm.generate([b["prompt"] for b in batch], batch_params, lora_request=lora_request)

                for item, output in zip(batch, outputs):
                    num_correct = 0
                    completions: List[str] = []
                    for candidate in output.outputs:
                        completions.append(candidate.text)
                        if math_verify_score(parse_fn, verify_fn, candidate.text, item["answer"]) > 0.5:
                            num_correct += 1
                    per_dataset[item["dataset"]]["total"] += 1
                    per_dataset[item["dataset"]]["correct_samples"] += num_correct
                    if num_correct > 0:
                        per_dataset[item["dataset"]]["correct"] += 1
                    sample_writer.write(json.dumps({
                        "run": run_idx,
                        "dataset": item["dataset"],
                        "index": item["index"],
                        "question": item["question"],
                        "ground_truth": item["answer"],
                        "num_correct": num_correct,
                        "completions": completions,
                    }, ensure_ascii=False) + "\n")

            run_summary: Dict[str, Any] = {
                "run": run_idx, "datasets": {},
                "overall": {"correct": 0, "correct_samples": 0, "total": 0, "pass_at_k": 0.0, "avg_at_k": 0.0},
            }
            for ds_name, counts in sorted(per_dataset.items()):
                correct = counts["correct"]
                correct_samples = counts["correct_samples"]
                total = counts["total"]
                total_samples = total * args.k
                run_summary["datasets"][ds_name] = {
                    "correct": correct,
                    "correct_samples": correct_samples,
                    "total": total,
                    "pass_at_k": correct / total if total else 0.0,
                    "avg_at_k": correct_samples / total_samples if total_samples else 0.0,
                    "k": args.k,
                }
                run_summary["overall"]["correct"] += correct
                run_summary["overall"]["correct_samples"] += correct_samples
                run_summary["overall"]["total"] += total
            ot = run_summary["overall"]["total"]
            ots = ot * args.k
            run_summary["overall"]["pass_at_k"] = run_summary["overall"]["correct"] / ot if ot else 0.0
            run_summary["overall"]["avg_at_k"] = run_summary["overall"]["correct_samples"] / ots if ots else 0.0
            final_results["runs"].append(run_summary)

    # Average across runs
    dataset_agg: Dict[str, Dict[str, float]] = {}
    for run in final_results["runs"]:
        for ds_name, metrics in run["datasets"].items():
            e = dataset_agg.setdefault(ds_name, {"correct": 0.0, "correct_samples": 0.0, "total": 0.0, "runs": 0.0})
            e["correct"] += metrics["correct"]
            e["correct_samples"] += metrics["correct_samples"]
            e["total"] += metrics["total"]
            e["runs"] += 1.0

    avg: Dict[str, Any] = {
        "datasets": {},
        "overall": {"correct": 0.0, "correct_samples": 0.0, "total": 0.0, "runs": repeats, "pass_at_k": 0.0, "avg_at_k": 0.0},
    }
    for ds_name, agg in sorted(dataset_agg.items()):
        ac = agg["correct"] / agg["runs"]
        acs = agg["correct_samples"] / agg["runs"]
        at = agg["total"] / agg["runs"]
        ats = at * args.k
        avg["datasets"][ds_name] = {
            "correct": ac,
            "correct_samples": acs,
            "total": at,
            "pass_at_k": ac / at if at else 0.0,
            "avg_at_k": acs / ats if ats else 0.0,
            "k": args.k,
            "runs": int(agg["runs"]),
        }
        avg["overall"]["correct"] += ac
        avg["overall"]["correct_samples"] += acs
        avg["overall"]["total"] += at
    ot = avg["overall"]["total"]
    ots = ot * args.k
    avg["overall"]["pass_at_k"] = avg["overall"]["correct"] / ot if ot else 0.0
    avg["overall"]["avg_at_k"] = avg["overall"]["correct_samples"] / ots if ots else 0.0
    final_results["average"] = avg

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(json.dumps(avg, indent=2, ensure_ascii=False))
    print(f"\nSummary → {out_json}")
    print(f"Samples  → {out_jsonl}")


if __name__ == "__main__":
    main()
