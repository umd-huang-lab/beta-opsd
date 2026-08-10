#!/usr/bin/env python
"""Compare student and OPSD self-teacher prompts on datasets with solutions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verify_boxed import extract_boxed_answer, grade_answer


STUDENT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

TEACHER_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["openthoughts_refs", "gsm8k"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no_thinking", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_reference_solution(refs: Any) -> str:
    if not isinstance(refs, list) or not refs:
        return ""
    first = refs[0]
    if isinstance(first, dict):
        return str(first.get("solution") or "").strip()
    return str(first or "").strip()


def load_examples(dataset: str, data_path: Path, limit: int, seed: int) -> list[dict[str, str]]:
    rows = read_jsonl(data_path)
    rng = random.Random(seed)

    examples: list[dict[str, str]] = []
    if dataset == "openthoughts_refs":
        for row in rows:
            solution = first_reference_solution(row.get("reference_solutions"))
            problem = str(row.get("problem") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if problem and answer and solution:
                examples.append(
                    {
                        "id": str(row.get("id") or len(examples)),
                        "question": problem,
                        "answer": answer,
                        "reference_solution": solution,
                    }
                )
    elif dataset == "gsm8k":
        for idx, row in enumerate(rows):
            question = str(row.get("question") or "").strip()
            solution = str(row.get("answer") or "").strip()
            answer = str(row.get("ground_truth") or "").strip()
            if question and answer and solution:
                examples.append(
                    {
                        "id": str(row.get("id") or idx),
                        "question": question,
                        "answer": answer,
                        "reference_solution": solution,
                    }
                )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    rng.shuffle(examples)
    return examples[:limit] if limit > 0 else examples


def build_messages(question: str, reference_solution: str, mode: str) -> list[dict[str, str]]:
    if mode == "student":
        return [{"role": "user", "content": f"Problem: {question}\n\n{STUDENT_INSTRUCTION}"}]
    if mode == "self_teacher":
        content = (
            f"Problem: {question}\n\n"
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{reference_solution}\n=== Reference Solution End ===\n"
            f"{TEACHER_TRANSITION_PROMPT}\n"
            f"{STUDENT_INSTRUCTION}"
        )
        return [{"role": "user", "content": content}]
    raise ValueError(f"Unsupported mode: {mode}")


def render_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]], thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def score_completion(completion: str, answer: str) -> bool:
    boxed = extract_boxed_answer(completion)
    if grade_answer(boxed or completion, answer):
        return True
    return grade_answer(completion, answer)


def summarize(samples: list[dict[str, Any]], k: int) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    for sample in samples:
        mode = sample["mode"]
        entry = totals.setdefault(mode, {"total": 0, "pass1": 0, "passk": 0, "correct_samples": 0})
        correct_flags = sample["correct"]
        entry["total"] += 1
        entry["pass1"] += int(bool(correct_flags[0]) if correct_flags else False)
        entry["passk"] += int(any(correct_flags))
        entry["correct_samples"] += sum(int(v) for v in correct_flags)

    metrics: dict[str, Any] = {}
    for mode, entry in sorted(totals.items()):
        total = entry["total"]
        total_samples = total * k
        metrics[mode] = {
            "total": total,
            "k": k,
            "pass_at_1": entry["pass1"] / total if total else 0.0,
            "pass_at_k": entry["passk"] / total if total else 0.0,
            "avg_at_k": entry["correct_samples"] / total_samples if total_samples else 0.0,
            "correct_first": entry["pass1"],
            "correct_any": entry["passk"],
            "correct_samples": entry["correct_samples"],
        }
    return metrics


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset, Path(args.data_path), args.limit, args.seed)
    if not examples:
        raise ValueError(f"No usable examples loaded from {args.data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    thinking = not args.no_thinking
    prompt_items: list[dict[str, Any]] = []
    for ex in examples:
        for mode in ("student", "self_teacher"):
            prompt_items.append(
                {
                    **ex,
                    "mode": mode,
                    "prompt": render_prompt(
                        tokenizer,
                        build_messages(ex["question"], ex["reference_solution"], mode),
                        thinking,
                    ),
                }
            )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )

    output_json = Path(args.output_json)
    output_jsonl = Path(args.output_jsonl)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as writer:
        for start in tqdm(range(0, len(prompt_items), args.batch_size), desc=args.dataset):
            batch = prompt_items[start : start + args.batch_size]
            outputs = llm.generate([item["prompt"] for item in batch], sampling_params=sampling_params)
            for item, output in zip(batch, outputs):
                completions = [candidate.text for candidate in output.outputs]
                correct = [score_completion(text, item["answer"]) for text in completions]
                record = {
                    "dataset": args.dataset,
                    "id": item["id"],
                    "mode": item["mode"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "correct": correct,
                    "num_correct": sum(int(v) for v in correct),
                    "completions": completions,
                }
                samples.append(record)
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "num_examples": len(examples),
        "model": args.model,
        "k": args.k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "thinking": thinking,
        "metrics": summarize(samples, args.k),
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
