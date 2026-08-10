import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from eval.verify_boxed import extract_boxed_answer, grade_answer


TEACHER_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


COPY_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, copy the reference solution exactly. "
    "Do not paraphrase it, do not derive your own solution, and do not add any extra text. "
    "Repeat the reference solution verbatim:\n"
)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_copy_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^<think>\s*", "", text)
    text = re.sub(r"\s*</think>\s*", " ", text)
    return normalize_ws(text)


def build_prompt(tokenizer, problem: str, solution: str, mode: str) -> tuple[str, bool]:
    if mode == "student":
        user_message = (
            f"Problem: {problem}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        messages = [{"role": "user", "content": user_message}]
        enable_thinking = False
    elif mode in ("teacher_independent", "teacher_training"):
        user_message = (
            f"Problem: {problem}\n\n"
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
            f"{TEACHER_TRANSITION_PROMPT}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        messages = [{"role": "user", "content": user_message}]
        enable_thinking = True
    elif mode == "teacher_copy":
        user_message = (
            f"Problem: {problem}\n\n"
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
            f"{COPY_TRANSITION_PROMPT}"
        )
        messages = [{"role": "user", "content": user_message}]
        enable_thinking = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt, enable_thinking


def load_examples(dataset_path: str, limit: int | None) -> list[dict]:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    examples = []
    for idx, row in enumerate(dataset):
        refs = row.get("reference_solutions", [])
        if not refs:
            continue
        ref0 = refs[0]
        solution = ref0["solution"] if isinstance(ref0, dict) else ref0
        gt_answer = row.get("answer")
        if gt_answer is None:
            gt_answer = extract_boxed_answer(solution) or solution
        examples.append(
            {
                "id": row.get("id", idx),
                "problem": row["problem"],
                "reference_solution": solution,
                "ground_truth_answer": gt_answer,
            }
        )
    return examples


def evaluate_mode(
    llm,
    tokenizer,
    examples: list[dict],
    mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    output_path: Path | None,
):
    prompts = []
    thinking_flags = []
    for ex in examples:
        prompt, enable_thinking = build_prompt(tokenizer, ex["problem"], ex["reference_solution"], mode)
        prompts.append(prompt)
        thinking_flags.append(enable_thinking)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    results = []
    exact_copy = 0
    normalized_copy = 0
    answer_correct = 0
    boxed_found = 0

    for ex, output, enable_thinking in zip(examples, outputs, thinking_flags):
        generated = output.outputs[0].text
        pred_answer = extract_boxed_answer(generated)
        is_answer_correct = grade_answer(pred_answer, ex["ground_truth_answer"])
        is_exact_copy = generated.strip() == ex["reference_solution"].strip()
        is_normalized_copy = normalize_copy_text(generated) == normalize_copy_text(ex["reference_solution"])

        exact_copy += int(is_exact_copy)
        normalized_copy += int(is_normalized_copy)
        answer_correct += int(is_answer_correct)
        boxed_found += int(pred_answer is not None)

        results.append(
            {
                "id": ex["id"],
                "mode": mode,
                "enable_thinking": enable_thinking,
                "ground_truth_answer": ex["ground_truth_answer"],
                "reference_boxed_answer": extract_boxed_answer(ex["reference_solution"]),
                "predicted_boxed_answer": pred_answer,
                "answer_correct": is_answer_correct,
                "exact_copy": is_exact_copy,
                "normalized_copy": is_normalized_copy,
                "reference_solution": ex["reference_solution"],
                "generation": generated,
            }
        )

    total = max(len(results), 1)
    summary = {
        "mode": mode,
        "num_examples": len(results),
        "answer_accuracy": answer_correct / total,
        "boxed_rate": boxed_found / total,
        "exact_copy_rate": exact_copy / total,
        "normalized_copy_rate": normalized_copy / total,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "results": results,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate student / teacher / copy prompt regimes.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_path", type=str, default="openthoughts_refs_1000_qwen3_8b.jsonl")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["student", "teacher_independent", "teacher_copy"],
        choices=["student", "teacher_independent", "teacher_training", "teacher_copy"],
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="analysis/results/copying_eval")
    args = parser.parse_args()

    examples = load_examples(args.dataset_path, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=32768,
        distributed_executor_backend="mp",
        enforce_eager=True,
    )

    run_summaries = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        output_path = output_dir / f"{mode}.json"
        summary = evaluate_mode(
            llm=llm,
            tokenizer=tokenizer,
            examples=examples,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            output_path=output_path,
        )
        run_summaries.append({k: v for k, v in summary.items() if k != "results"})

    combined = {
        "base_model": args.base_model,
        "dataset_path": args.dataset_path,
        "limit": len(examples),
        "modes": run_summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
