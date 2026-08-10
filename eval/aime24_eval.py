"""
AIME 2024 benchmark evaluation during training (vLLM colocate only).

Uses the same grading utilities as eval/evaluate_math.py.

Sampling defaults are aligned with eval/evaluate_math.py (standalone math eval).
Keep the constants below in sync when you change evaluate_math.py argparse defaults.

NOTE: max_new_tokens is large (38912). Ensure training --max_length (vLLM max_model_len)
is >= prompt tokens + max_new_tokens or generation may fail or truncate.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from transformers import TrainerCallback, TrainerControl, TrainerState

from eval.verify_boxed import extract_boxed_answer, grade_answer

# ---------------------------------------------------------------------------
# Defaults — match eval/evaluate_math.py (see evaluate_math.py `main()` argparse)
# ---------------------------------------------------------------------------
AIME24_EVAL_TEMPERATURE = 1.0
AIME24_EVAL_MAX_NEW_TOKENS = 38912
AIME24_EVAL_TOP_P = 1.0  # 1.0 = no nucleus restriction (same as evaluate_math default)
AIME24_EVAL_TOP_K = -1  # disabled
AIME24_EVAL_MIN_P = 0.0
AIME24_EVAL_PRESENCE_PENALTY = 0.0
AIME24_EVAL_VAL_N = 1  # pass@1 by default (set env AIME24_EVAL_VAL_N to increase k)
AIME24_EVAL_ENABLE_THINKING = True
# vLLM default; evaluate_math SamplingParams does not set this explicitly
AIME24_EVAL_REPETITION_PENALTY = 1.0


def _env_override(name: str, default: Any, cast):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except Exception:
        return default


def _resolved_eval_settings(trainer: Any) -> dict[str, Any]:
    """
    Build hyperparameters for this eval.

    Resolution order per field: env var (AIME24_EVAL_*) → optional trainer.args.aime24_eval_* → module constant.

    Env vars: AIME24_EVAL_TEMPERATURE, AIME24_EVAL_MAX_NEW_TOKENS, AIME24_EVAL_TOP_P,
    AIME24_EVAL_TOP_K, AIME24_EVAL_MIN_P, AIME24_EVAL_PRESENCE_PENALTY, AIME24_EVAL_VAL_N,
    AIME24_EVAL_ENABLE_THINKING (0 or 1), AIME24_EVAL_REPETITION_PENALTY.
    """
    args = getattr(trainer, "args", None)

    def from_args(attr: str, default: Any):
        if args is None:
            return default
        v = getattr(args, attr, None)
        return default if v is None else v

    if os.environ.get("AIME24_EVAL_ENABLE_THINKING") is not None:
        enable_thinking = bool(int(os.environ["AIME24_EVAL_ENABLE_THINKING"]))
    else:
        enable_thinking = bool(from_args("aime24_eval_enable_thinking", AIME24_EVAL_ENABLE_THINKING))

    return {
        "temperature": _env_override(
            "AIME24_EVAL_TEMPERATURE",
            from_args("aime24_eval_temperature", AIME24_EVAL_TEMPERATURE),
            float,
        ),
        "max_new_tokens": int(
            _env_override(
                "AIME24_EVAL_MAX_NEW_TOKENS",
                from_args("aime24_eval_max_new_tokens", AIME24_EVAL_MAX_NEW_TOKENS),
                int,
            )
        ),
        "top_p": _env_override(
            "AIME24_EVAL_TOP_P",
            from_args("aime24_eval_top_p", AIME24_EVAL_TOP_P),
            float,
        ),
        "top_k": int(
            _env_override(
                "AIME24_EVAL_TOP_K",
                from_args("aime24_eval_top_k", AIME24_EVAL_TOP_K),
                int,
            )
        ),
        "min_p": _env_override(
            "AIME24_EVAL_MIN_P",
            from_args("aime24_eval_min_p", AIME24_EVAL_MIN_P),
            float,
        ),
        "presence_penalty": _env_override(
            "AIME24_EVAL_PRESENCE_PENALTY",
            from_args("aime24_eval_presence_penalty", AIME24_EVAL_PRESENCE_PENALTY),
            float,
        ),
        "val_n": int(
            _env_override(
                "AIME24_EVAL_VAL_N",
                from_args("aime24_eval_val_n", AIME24_EVAL_VAL_N),
                int,
            )
        ),
        "enable_thinking": enable_thinking,
        "repetition_penalty": float(
            _env_override(
                "AIME24_EVAL_REPETITION_PENALTY",
                from_args("aime24_eval_repetition_penalty", AIME24_EVAL_REPETITION_PENALTY),
                float,
            )
        ),
    }


def _log_aime24_metrics(trainer: Any, accuracy: float, correct: int, global_step: int) -> None:
    """
    Log AIME metrics: Trainer.log (TensorBoard / whatever report_to lists), plus W&B.

    If `training_args.report_to` does not include \"wandb\", HuggingFace will not send these to W&B
    even though opsd_train.py may have called wandb.init() manually — in that case we call wandb.log
    ourselves. If report_to already includes wandb, Trainer.log is enough (avoid double-logging).
    """
    if not getattr(trainer.accelerator, "is_main_process", True):
        return

    step = int(getattr(trainer.state, "global_step", global_step))
    metrics = {
        "eval/aime24_accuracy": float(accuracy),
        "eval/aime24_correct": float(correct),
    }
    try:
        trainer.log(metrics)
    except Exception as exc:
        print(f"[AIME24 eval] trainer.log failed: {exc!r}")

    report_to = getattr(trainer.args, "report_to", None) or []
    if isinstance(report_to, str):
        report_to = [report_to]

    try:
        import wandb

        run = getattr(wandb, "run", None)
        # Manual init in opsd_train without report_to=wandb → HF does not forward trainer.log to W&B
        if "wandb" not in report_to and run is not None:
            wandb.log(metrics, step=step, commit=True)
            run.summary["aime24_accuracy"] = float(accuracy)
            run.summary["aime24_correct"] = int(correct)
            run.summary["aime24_step"] = step
        elif "wandb" not in report_to and run is None:
            print(
                "[AIME24 eval] W&B: wandb.run is None and report_to does not include 'wandb' — "
                "metrics only went to trainer.log (e.g. TensorBoard). "
                "Fix: pass --report_to wandb in your training config, or ensure wandb.init() "
                "succeeds on rank 0 (WANDB_API_KEY, LOCAL_RANK=0)."
            )
        elif "wandb" in report_to and run is not None:
            run.summary["aime24_accuracy"] = float(accuracy)
            run.summary["aime24_correct"] = int(correct)
            run.summary["aime24_step"] = step
    except Exception as exc:
        print(f"[AIME24 eval] wandb.log / summary failed: {exc!r}")


def run_aime24_evaluation(trainer: Any, global_step: int) -> Optional[dict]:
    """
    Evaluate current student weights on HuggingFaceH4/aime_2024 (train split).
    Syncs weights to vLLM, runs batch generation, saves JSON under output_dir/eval/.

    Metrics: pass@k accuracy — a problem counts as correct if any of k samples matches (same spirit as evaluate_math val_n).
    """
    if not trainer.accelerator.is_main_process:
        return None

    if not getattr(trainer, "use_vllm", False):
        print("[AIME24 eval] Skipping: use_vllm=False (hook requires vLLM).")
        return None

    vllm_mode = getattr(trainer, "vllm_mode", None)
    if vllm_mode != "colocate":
        print(f"[AIME24 eval] Skipping: vllm_mode={vllm_mode!r} (only colocate is supported).")
        return None

    from vllm import SamplingParams

    cfg = _resolved_eval_settings(trainer)
    temperature = cfg["temperature"]
    max_new_tokens = cfg["max_new_tokens"]
    top_p = cfg["top_p"]
    top_k = cfg["top_k"]
    min_p = cfg["min_p"]
    presence_penalty = cfg["presence_penalty"]
    val_n = max(1, int(cfg["val_n"]))
    enable_thinking = bool(cfg["enable_thinking"])
    repetition_penalty = cfg["repetition_penalty"]

    print(f"\n{'=' * 80}\nAIME 2024 evaluation at global_step={global_step}\n{'=' * 80}")
    print(
        "AIME24 eval settings (aligned with evaluate_math.py defaults; see eval/aime24_eval.py): "
        f"temperature={temperature}, max_new_tokens={max_new_tokens}, top_p={top_p}, top_k={top_k}, "
        f"min_p={min_p}, presence_penalty={presence_penalty}, val_n={val_n}, enable_thinking={enable_thinking}"
    )
    train_max_len = getattr(getattr(trainer, "args", None), "max_length", None)
    if train_max_len is not None and int(train_max_len) < max_new_tokens + 4096:
        print(
            f"[AIME24 eval] WARNING: training max_length={train_max_len} is likely too small for "
            f"prompt + max_new_tokens={max_new_tokens}. Raise --max_length (vLLM max_model_len) or lower "
            f"AIME24_EVAL_MAX_NEW_TOKENS."
        )

    trainer._wake_vllm_if_needed()
    trainer._move_model_to_vllm()

    tokenizer = trainer.processing_class

    guided_decoding = None
    if getattr(trainer, "vllm_guided_decoding_regex", None):
        from vllm.sampling_params import StructuredOutputsParams

        guided_decoding = StructuredOutputsParams(
            regex=trainer.vllm_guided_decoding_regex
        )

    sampling_params = SamplingParams(
        n=val_n,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        guided_decoding=guided_decoding,
    )

    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    all_prompts: list[str] = []
    all_gt: list[str] = []
    for example in dataset:
        problem = example["problem"]
        gt_answer = example["answer"]
        # Match evaluate_math-style user message; thinking follows tokenizer template
        user_message = (
            f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )
        messages = [{"role": "user", "content": user_message}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        all_prompts.append(text)
        all_gt.append(str(gt_answer))

    all_outputs = trainer.vllm_engine.generate(
        all_prompts, sampling_params=sampling_params, use_tqdm=True
    )

    correct = 0
    results: list[dict] = []
    for idx, (output, gt) in enumerate(zip(all_outputs, all_gt)):
        sample_results = []
        any_correct = False
        for rank, o in enumerate(output.outputs):
            generated_text = o.text
            pred = extract_boxed_answer(generated_text)
            ok = grade_answer(pred, gt) if pred is not None else False
            if ok:
                any_correct = True
            sample_results.append(
                {
                    "sample_i": rank,
                    "predicted_boxed": pred,
                    "is_correct": ok,
                    "completion": generated_text,
                }
            )
        if any_correct:
            correct += 1
        results.append(
            {
                "idx": idx,
                "ground_truth": gt,
                "is_correct_pass_at_k": any_correct,
                "val_n": val_n,
                "samples": sample_results,
            }
        )

    n = len(all_gt)
    accuracy = correct / n if n else 0.0
    summary = {
        "global_step": global_step,
        "dataset": "HuggingFaceH4/aime_2024",
        "num_problems": n,
        "num_correct_pass_at_k": correct,
        "accuracy_pass_at_k": accuracy,
        "val_n": val_n,
        "eval_settings": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "enable_thinking": enable_thinking,
            "repetition_penalty": repetition_penalty,
        },
        "results": results,
    }

    out_dir = Path(trainer.args.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"aime24_step_{global_step}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"AIME24 pass@{val_n} accuracy: {accuracy * 100:.2f}% ({correct}/{n} problems with ≥1 correct sample)"
    )
    print(f"Saved: {out_path}\n{'=' * 80}\n")

    _log_aime24_metrics(trainer, accuracy, correct, global_step)

    if getattr(trainer, "vllm_enable_sleep_mode", False):
        try:
            trainer.vllm_engine.sleep(level=2)
        except Exception:
            pass

    return summary


class AIME24EvalCallback(TrainerCallback):
    """
    Run AIME 2024 eval every `interval` global steps (25 → steps 25, 50, 75, …).

    Note: `save_steps` only controls checkpoint saves; it does not trigger this callback.
    """

    def __init__(self, trainer: Any, interval: int = 25):
        self.trainer = trainer
        self.interval = max(1, int(interval))

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        gs = int(state.global_step)
        if gs < self.interval or gs % self.interval != 0:
            return control
        if self.trainer.accelerator.is_main_process:
            run_aime24_evaluation(self.trainer, gs)
        self.trainer.accelerator.wait_for_everyone()
        return control
