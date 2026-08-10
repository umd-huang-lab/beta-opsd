import os
import wandb

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold.gold_config import GOLDConfig

from beta_opsd_trainer import BetaOPSDTrainer

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Arguments for beta-OPSD training."""

    dataset_path: str = field(
        default="",
        metadata={
            "help": "Optional local dataset path. If set, loads this file via datasets.json instead of the default HF dataset. "
            "Rows must contain `problem` plus either `solution` or `reference_solutions`."
        },
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to load when using --dataset_path."},
    )
    use_tinker_loss: bool = field(
        default=True,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of the default full-vocab JSD loss. "
            "This is the default beta-OPSD loss."
        },
    )
    tinker_use_reward_to_go: bool = field(
        default=False,
        metadata={
            "help": "When use_tinker_loss=True, replace the per-token teacher-student reward with its suffix-sum "
            "reward-to-go before weighting student log-probs."
        },
    )
    tinker_reward_to_go_discount: float = field(
        default=1.0,
        metadata={
            "help": "Discount factor gamma for tinker reward-to-go. gamma=1.0 recovers the undiscounted suffix sum; "
            "gamma<1.0 exponentially downweights later rewards."
        },
    )
    tinker_advantage_clip: float = field(
        default=0.0,
        metadata={
            "help": "Symmetrically clamp the detached Tinker advantage after optional reward-to-go. "
            "Set <=0 to disable. Example: 3 clamps to [-3, 3]."
        },
    )
    use_mixed_teacher_target: bool = field(
        default=True,
        metadata={
            "help": "For Tinker loss, use a target distribution that mixes the frozen initial student "
            "on the student prompt with the teacher on the privileged teacher prompt. Enabled by default; "
            "it is disabled automatically when --use_mixed_sampling is set."
        },
    )
    disable_mixed_teacher_target: bool = field(
        default=False,
        metadata={
            "help": "Disable the default mixed teacher target and use the teacher distribution directly. "
            "This is mainly for fixed-teacher baselines."
        },
    )
    mixed_teacher_target_teacher_weight: float = field(
        default=0.5,
        metadata={
            "help": "Teacher weight in the mixed Tinker target: "
            "(1-teacher_weight)*reference_student + teacher_weight*teacher."
        },
    )
    mixed_teacher_target_teacher_weight_linear_decay: bool = field(
        default=False,
        metadata={
            "help": "Linearly schedule mixed_teacher_target_teacher_weight to "
            "mixed_teacher_target_teacher_weight_final over training."
        },
    )
    mixed_teacher_target_teacher_weight_final: float = field(
        default=0.5,
        metadata={
            "help": "Final teacher target weight when "
            "mixed_teacher_target_teacher_weight_linear_decay=True."
        },
    )
    mixed_teacher_target_reference_model: str = field(
        default="frozen_reference",
        metadata={
            "help": "Reference side of the mixed Tinker target. Use 'frozen_reference' for the adapter-disabled "
            "initial student, or 'current_student' for detached current student log-probs."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    use_mixed_sampling: bool = field(
        default=False,
        metadata={
            "help": "Sample each generation token from a convex mixture of student and teacher "
            "distributions: (1-alpha)*q_student + alpha*p_teacher. Requires mixed_sampling_teacher_alpha > 0. "
            "When enabled, the distillation target is the fixed teacher rather than the mixed teacher target. "
            "With use_vllm (colocate mode), uses the modified vLLM shadow-request approach. "
            "Without use_vllm, uses the HF generate path with KV-cached teacher."
        },
    )
    mixed_sampling_teacher_alpha: float = field(
        default=0.0,
        metadata={
            "help": "Teacher mixture weight in [0, 1]. 0 = pure student, 1 = pure teacher. "
            "Only used when use_mixed_sampling=True."
        },
    )
    mixed_sampling_teacher_alpha_linear_decay: bool = field(
        default=False,
        metadata={
            "help": "If set, linearly decay the teacher mixture weight from "
            "mixed_sampling_teacher_alpha to mixed_sampling_teacher_alpha_final over training."
        },
    )
    mixed_sampling_teacher_alpha_exponential_decay: bool = field(
        default=False,
        metadata={
            "help": "If set, exponentially decay the teacher mixture weight from "
            "mixed_sampling_teacher_alpha to mixed_sampling_teacher_alpha_final over training."
        },
    )
    mixed_sampling_teacher_alpha_final: float = field(
        default=0.0,
        metadata={
            "help": "Final teacher mixture weight when mixed_sampling_teacher_alpha_linear_decay=True. "
            "Ignored otherwise."
        },
    )
    use_per_token_importance_sampling: bool = field(
        default=False,
        metadata={
            "help": "Apply per-token importance weights student_prob/proposal_prob for tinker losses when "
            "samples come from a mixed proposal distribution. Supported for HF generation and vLLM colocate "
            "mixed sampling; not supported for vLLM server mode."
        },
    )
    importance_sampling_clip_min: float = field(
        default=0.0,
        metadata={
            "help": "Optional lower clamp for per-token importance weights. Set <=0 to disable the lower bound."
        },
    )
    importance_sampling_clip_max: float = field(
        default=0.0,
        metadata={
            "help": "Optional upper clamp for per-token importance weights. Set <=0 to disable the upper bound."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.disable_mixed_teacher_target:
        script_args.use_mixed_teacher_target = False

    if script_args.use_mixed_sampling:
        if script_args.use_ema_teacher:
            raise ValueError("use_mixed_sampling=True requires a fixed teacher and cannot be combined with use_ema_teacher.")
        script_args.fixed_teacher = True
        script_args.use_mixed_teacher_target = False

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    _dtype_str = getattr(model_args, "dtype", "bfloat16") or "bfloat16"
    loss_name = "tinker" if script_args.use_tinker_loss else "jsd"
    if script_args.use_tinker_loss and script_args.tinker_use_reward_to_go:
        loss_name = "tinker_rtg"
    teacher_strategy = "ema" if script_args.use_ema_teacher else "fixed" if script_args.fixed_teacher else "student"
    if script_args.use_mixed_sampling:
        if script_args.mixed_sampling_teacher_alpha_linear_decay:
            alpha_schedule = (
                f"linear_decay:{script_args.mixed_sampling_teacher_alpha}"
                f"->{script_args.mixed_sampling_teacher_alpha_final}"
            )
        elif script_args.mixed_sampling_teacher_alpha_exponential_decay:
            alpha_schedule = (
                f"exponential_decay:{script_args.mixed_sampling_teacher_alpha}"
                f"->{script_args.mixed_sampling_teacher_alpha_final}"
            )
        else:
            alpha_schedule = f"constant:{script_args.mixed_sampling_teacher_alpha}"
    else:
        alpha_schedule = "disabled"

    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_config = (
            f"opsd_mix_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    print(f"\n{'='*80}")
    print("RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Loss: {loss_name}")
    print(f"Teacher strategy: {teacher_strategy}")
    print(f"Effective batch size: {effective_batch_size}")
    print(
        "vLLM: "
        f"enabled={training_args.use_vllm}, "
        f"mode={getattr(training_args, 'vllm_mode', 'disabled')}, "
        f"tp={getattr(training_args, 'vllm_tensor_parallel_size', 'n/a')}, "
        f"mem_util={getattr(training_args, 'vllm_gpu_memory_utilization', 'n/a')}"
    )
    print(
        "Sampling: "
        f"temperature={training_args.temperature}, top_p={training_args.top_p}, top_k={training_args.top_k}"
    )
    print(
        "Sequence limits: "
        f"max_length={training_args.max_length}, max_completion_length={training_args.max_completion_length}"
    )
    print(
        "Mixture sampling: "
        f"enabled={script_args.use_mixed_sampling}, schedule={alpha_schedule}, space=pre_softmax"
    )
    print(
        "Distillation controls: "
        f"jsd_token_clip={script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else 'off'}, "
        f"top_k_loss={script_args.top_k_loss if script_args.top_k_loss > 0 else 'full_vocab'}, "
        f"tinker_advantage_clip={script_args.tinker_advantage_clip if script_args.tinker_advantage_clip > 0 else 'off'}, "
        f"mixed_teacher_target={script_args.use_mixed_teacher_target}, "
        f"target_teacher_weight={script_args.mixed_teacher_target_teacher_weight}, "
        f"target_teacher_weight_final={script_args.mixed_teacher_target_teacher_weight_final}, "
        f"target_teacher_weight_linear={script_args.mixed_teacher_target_teacher_weight_linear_decay}, "
        f"target_reference={script_args.mixed_teacher_target_reference_model}"
    )
    print(
        "Importance sampling: "
        f"enabled={script_args.use_per_token_importance_sampling}, "
        f"clip_min={script_args.importance_sampling_clip_min if script_args.importance_sampling_clip_min > 0 else 'off'}, "
        f"clip_max={script_args.importance_sampling_clip_max if script_args.importance_sampling_clip_max > 0 else 'off'}"
    )
    print(f"{'='*80}\n")

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "loss_name": loss_name,
                "teacher_strategy": teacher_strategy,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "max_steps": training_args.max_steps,
                "temperature": training_args.temperature,
                "top_p": training_args.top_p,
                "top_k": training_args.top_k,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "attn_implementation": model_args.attn_implementation or "flash_attention_2",
                "dtype": _dtype_str,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "fixed_teacher": script_args.fixed_teacher,
                "use_tinker_loss": script_args.use_tinker_loss,
                "tinker_use_reward_to_go": script_args.tinker_use_reward_to_go if script_args.use_tinker_loss else None,
                "tinker_reward_to_go_discount": script_args.tinker_reward_to_go_discount
                if script_args.use_tinker_loss and script_args.tinker_use_reward_to_go
                else None,
                "tinker_advantage_clip": script_args.tinker_advantage_clip
                if script_args.use_tinker_loss and script_args.tinker_advantage_clip > 0
                else None,
                "use_mixed_teacher_target": script_args.use_mixed_teacher_target
                if script_args.use_tinker_loss
                else None,
                "mixed_teacher_target_teacher_weight": script_args.mixed_teacher_target_teacher_weight
                if script_args.use_tinker_loss and script_args.use_mixed_teacher_target
                else None,
                "mixed_teacher_target_teacher_weight_linear_decay": script_args.mixed_teacher_target_teacher_weight_linear_decay
                if script_args.use_tinker_loss and script_args.use_mixed_teacher_target
                else None,
                "mixed_teacher_target_teacher_weight_final": script_args.mixed_teacher_target_teacher_weight_final
                if script_args.use_tinker_loss and script_args.use_mixed_teacher_target
                else None,
                "mixed_teacher_target_reference_model": script_args.mixed_teacher_target_reference_model
                if script_args.use_tinker_loss and script_args.use_mixed_teacher_target
                else None,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "jsd_token_clip": script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
                "jsd_token_clip_active": not script_args.use_tinker_loss and script_args.jsd_token_clip > 0,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "use_vllm": training_args.use_vllm,
                "vllm_mode": getattr(training_args, "vllm_mode", None),
                "vllm_gpu_memory_utilization": getattr(training_args, "vllm_gpu_memory_utilization", None),
                "vllm_tensor_parallel_size": getattr(training_args, "vllm_tensor_parallel_size", None),
                "use_mixed_sampling": script_args.use_mixed_sampling,
                "mixed_sampling_space": "pre_softmax" if script_args.use_mixed_sampling else None,
                "mixed_sampling_teacher_alpha": script_args.mixed_sampling_teacher_alpha
                if script_args.use_mixed_sampling
                else None,
                "mixed_sampling_teacher_alpha_linear_decay": script_args.mixed_sampling_teacher_alpha_linear_decay
                if script_args.use_mixed_sampling
                else None,
                "mixed_sampling_teacher_alpha_exponential_decay": script_args.mixed_sampling_teacher_alpha_exponential_decay
                if script_args.use_mixed_sampling
                else None,
                "mixed_sampling_teacher_alpha_schedule": alpha_schedule if script_args.use_mixed_sampling else None,
                "mixed_sampling_teacher_alpha_final": script_args.mixed_sampling_teacher_alpha_final
                if script_args.use_mixed_sampling
                and (
                    script_args.mixed_sampling_teacher_alpha_linear_decay
                    or script_args.mixed_sampling_teacher_alpha_exponential_decay
                )
                else None,
                "use_per_token_importance_sampling": script_args.use_per_token_importance_sampling,
                "importance_sampling_clip_min": script_args.importance_sampling_clip_min
                if script_args.importance_sampling_clip_min > 0
                else None,
                "importance_sampling_clip_max": script_args.importance_sampling_clip_max
                if script_args.importance_sampling_clip_max > 0
                else None,
            },
        )

    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": torch.bfloat16,
    }
    model_dtype = dtype_map.get(_dtype_str.lower(), torch.bfloat16)

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args.presence_penalty = script_args.presence_penalty

    if script_args.dataset_path:
        dataset = load_dataset("json", data_files=script_args.dataset_path, split=script_args.dataset_split)

        def _to_single_teacher_example(example):
            if example.get("solution"):
                return {"problem": example["problem"], "solution": example["solution"]}

            refs = example.get("reference_solutions", [])
            if isinstance(refs, list) and refs and isinstance(refs[0], dict) and refs[0].get("solution"):
                return {"problem": example["problem"], "solution": refs[0]["solution"]}

            raise ValueError(
                "Local dataset rows must contain `solution` or a non-empty `reference_solutions` list with `solution` fields."
            )

        train_dataset = dataset.map(_to_single_teacher_example)
    else:
        dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
        train_dataset = dataset["train"]

    trainer = BetaOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        tinker_use_reward_to_go=script_args.tinker_use_reward_to_go,
        tinker_reward_to_go_discount=script_args.tinker_reward_to_go_discount,
        tinker_advantage_clip=script_args.tinker_advantage_clip
        if script_args.tinker_advantage_clip > 0
        else None,
        use_mixed_teacher_target=script_args.use_mixed_teacher_target,
        mixed_teacher_target_teacher_weight=script_args.mixed_teacher_target_teacher_weight,
        mixed_teacher_target_teacher_weight_linear_decay=script_args.mixed_teacher_target_teacher_weight_linear_decay,
        mixed_teacher_target_teacher_weight_final=script_args.mixed_teacher_target_teacher_weight_final,
        mixed_teacher_target_reference_model=script_args.mixed_teacher_target_reference_model,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        use_mixed_sampling=script_args.use_mixed_sampling,
        mixed_sampling_teacher_alpha=script_args.mixed_sampling_teacher_alpha,
        mixed_sampling_teacher_alpha_linear_decay=script_args.mixed_sampling_teacher_alpha_linear_decay,
        mixed_sampling_teacher_alpha_exponential_decay=script_args.mixed_sampling_teacher_alpha_exponential_decay,
        mixed_sampling_teacher_alpha_final=script_args.mixed_sampling_teacher_alpha_final,
        use_per_token_importance_sampling=script_args.use_per_token_importance_sampling,
        importance_sampling_clip_min=script_args.importance_sampling_clip_min
        if script_args.importance_sampling_clip_min > 0
        else None,
        importance_sampling_clip_max=script_args.importance_sampling_clip_max
        if script_args.importance_sampling_clip_max > 0
        else None,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)
