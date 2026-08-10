# Beta-OPSD

Beta-OPSD is the code release for experiments around beta on-policy self-distillation. The main implementation trains a student on its own on-policy samples while using a target distribution that interpolates between a reference student and a privileged teacher. The repo also includes mixed-sampling ablations, fixed-teacher OPSD baselines, SFT/GRPO baselines, and math evaluation scripts.

This repository is intentionally clean: it contains source code, launch configs, evaluation inputs, and documentation only. Training outputs, checkpoints, W&B state, Slurm logs, generated samples, caches, and large generated reference files are excluded by `.gitignore`.

Paper: [arXiv:2607.28582](https://arxiv.org/abs/2607.28582). The PDF is included at [`docs/beta-opsd-paper.pdf`](docs/beta-opsd-paper.pdf).

## Installation

```bash
conda env create -f environment.yml
conda activate opsd
```

Install FlashAttention after the environment is created:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

Key dependency versions are pinned in [`environment.yml`](environment.yml), including PyTorch 2.8.0, Transformers 4.57.1, TRL 0.26.0, vLLM 0.11.0, PEFT 0.17.1, and DeepSpeed 0.18.2.

## Repository Layout

```text
.
├── beta_opsd_train.py          # Main beta-OPSD training entry point
├── beta_opsd_trainer.py        # BetaOPSDTrainer: mix-target and mixed-sampling logic
├── data_collator.py            # Self-distillation data collator
├── opsd_train.py               # Fixed-teacher/default OPSD baseline entry point
├── opsd_trainer.py             # Baseline OPSD trainer
├── sft_train.py                # SFT baseline
├── grpo_train.py               # GRPO baseline
├── accelerate.yaml             # Multi-GPU DeepSpeed/Accelerate config
├── accelerate_smoke.yaml       # Single-GPU smoke-test config
├── scripts/                    # Slurm and shell launchers
├── eval/                       # Math evaluation code and benchmark JSONL files
├── analysis/                   # Auxiliary analysis scripts
└── docs/                       # Paper PDF
```

The beta-OPSD files are the default place to start:

- [`beta_opsd_train.py`](beta_opsd_train.py): CLI arguments, model/dataset setup, trainer construction.
- [`beta_opsd_trainer.py`](beta_opsd_trainer.py): target interpolation, Tinker-style sampled-token loss, optional mixed sampling, and optional reward-to-go.

The `scripts/` folder is limited to paper-relevant launchers and smoke tests:

| Script family | Purpose |
|---|---|
| `beta_opsd_mix_target_qwen34b_h200.sbatch` | Main beta-OPSD mixed-target run. |
| `beta_opsd_mixed_sampling*.sbatch` | Mixed-sampling proposal ablations and RTG variants. |
| `beta_opsd_fixed_teacher_qwen34b_h200.sbatch`, `submit_opsd_*_localvllm.sbatch`, `run_opsd_*.sh` | Vanilla/fixed-teacher OPSD baselines. |
| `run_sft.sh`, `run_grpo.sh`, `submit_qwen*_baseline.sbatch` | SFT and GRPO baselines at paper model scales. |
| `smoke*.sbatch`, `smoke_test.sh` | Short sanity checks for the released methods. |
| `serve_vllm.sh`, `run_with_local_vllm.sh`, `submit_vllm_server.sbatch` | vLLM helpers used by the OPSD launchers. |

## Core Method Switches

Beta-OPSD defaults are set in [`beta_opsd_train.py`](beta_opsd_train.py):

| Argument | Default | Meaning |
|---|---:|---|
| `--use_tinker_loss` | `True` | Use sampled-token reverse-KL style beta-OPSD loss. |
| `--use_mixed_teacher_target` | `True` | Use the interpolated target distribution. |
| `--mixed_teacher_target_teacher_weight` | `0.5` | Weight of teacher logits in the target interpolation. |
| `--mixed_teacher_target_reference_model` | `frozen_reference` | Reference side of the target: `frozen_reference` or `current_student`. |
| `--mixed_teacher_target_teacher_weight_linear_decay` | `False` | Linearly schedule teacher weight over training. |
| `--mixed_teacher_target_teacher_weight_final` | `0.5` | Final teacher weight for the linear schedule. |
| `--use_mixed_sampling` | `False` | Sample from a student/teacher proposal mixture. When enabled, target is forced to fixed teacher. |
| `--disable_mixed_teacher_target` | `False` | Disable the default mix-target behavior for fixed-teacher baselines. |
| `--tinker_use_reward_to_go` | `False` | Use reward-to-go weighting for the sampled-token advantage. |
| `--use_per_token_importance_sampling` | `False` | Apply proposal correction for mixed-sampling runs. |

Target interpolation uses:

```text
target = (1 - teacher_weight) * reference_student + teacher_weight * privileged_teacher
```

`teacher_weight` is the weight of teacher logits/probabilities in this interpolation. For mixed sampling, `--use_mixed_sampling` is an ablation/proposal option; the code automatically uses a fixed-teacher target instead of the mixed teacher target.

## Quick Smoke Tests

Smoke tests are short Slurm jobs intended to catch import, config, trainer, and basic GPU/runtime issues.

```bash
sbatch scripts/smoke_test.sh                         # OPSD baseline
sbatch scripts/beta_opsd_smoke_mix_target.sbatch     # beta-OPSD mix target
sbatch scripts/beta_opsd_smoke_mixed_sampling.sbatch # beta-OPSD mixed sampling
sbatch scripts/smoke_sft.sbatch                      # SFT baseline
sbatch scripts/smoke_grpo.sbatch                     # GRPO baseline
```

Logs are written to:

```text
slurm_logs/
runs/<run_name>/logs/
```

Both directories are ignored by Git.

## Training Recipes

### Beta-OPSD Mix Target

The primary beta-OPSD configuration uses a mixed teacher target. Example:

```bash
RUN=beta_target_qwen34b \
MODEL_NAME_OR_PATH=Qwen/Qwen3-4B \
TEACHER_WEIGHT_START=0.5 \
MAX_STEPS=200 \
sbatch scripts/beta_opsd_mix_target_qwen34b_h200.sbatch
```

For a linearly scheduled target teacher weight:

```bash
RUN=beta_target_qwen34b_05to08 \
TEACHER_WEIGHT_START=0.5 \
TEACHER_WEIGHT_FINAL=0.8 \
ALPHA_SCHEDULE=linear \
sbatch scripts/beta_opsd_mix_target_qwen34b_h200.sbatch
```

### Mixed-Sampling Ablation

Mixed sampling changes the proposal distribution used for on-policy generation. It is not the default beta-OPSD target. When `--use_mixed_sampling` is enabled, the training target is forced to fixed teacher.

```bash
RUN=beta_mixed_sampling_qwen17b \
MIXED_ALPHA_START=0.8 \
MIXED_ALPHA_FINAL=0.2 \
sbatch scripts/beta_opsd_mixed_sampling.sbatch
```

Reward-to-go and importance-sampling variants are provided:

```bash
sbatch scripts/beta_opsd_mixed_sampling_rtg.sbatch
sbatch scripts/beta_opsd_mixed_sampling_fixed_alpha_rtg.sbatch
```

### Fixed-Teacher OPSD Baselines

The original OPSD-style baseline entry point is kept for comparison:

```bash
RUN=opsd_qwen17b sbatch scripts/submit_opsd_1b_localvllm.sbatch
RUN=opsd_qwen34b sbatch scripts/beta_opsd_fixed_teacher_qwen34b_h200.sbatch
```

### SFT and GRPO Baselines

```bash
RUN=sft_qwen34b bash scripts/run_sft.sh
RUN=grpo_qwen34b bash scripts/run_grpo.sh
```

Slurm templates for Qwen3-1.7B and Qwen3-8B baseline runs are also included under `scripts/submit_qwen*_baseline.sbatch`.

## Evaluation

`eval/eval_math.py` evaluates base models and LoRA checkpoints on math benchmarks using vLLM. Included benchmark files live in [`eval/data`](eval/data).

Evaluate a run checkpoint:

```bash
RUN=beta_target_qwen34b STEP=100 sbatch eval/submit.sh
```

Evaluate a base model:

```bash
BASE_MODEL=Qwen/Qwen3-1.7B sbatch eval/submit.sh
```

Useful eval environment variables:

| Variable | Meaning |
|---|---|
| `RUN` | Run name under `runs/`. |
| `STEP` | Checkpoint step, resolved as `runs/<RUN>/checkpoints/checkpoint-<STEP>`. |
| `CHECKPOINT` | Explicit checkpoint path override. |
| `BASE_MODEL` | Base model to evaluate when no checkpoint is given. |
| `DATASETS` | Comma-separated dataset list. |
| `DECODING` | `greedy` or `sampling`. |
| `K` | Number of samples per problem for pass@k/avg@k style evaluation. |
| `THINKING` | Set to `1` for Qwen3 thinking mode. |

Generated evaluation JSON/JSONL outputs are ignored by Git.

## Data

Training defaults to the Hugging Face dataset:

```text
siyanzhao/Openthoughts_math_30k_opsd
```

`beta_opsd_train.py` also accepts a local JSON/JSONL dataset:

```bash
python beta_opsd_train.py --dataset_path path/to/data.jsonl
```

Rows should contain `problem` and either `solution` or `reference_solutions`. When `reference_solutions` is present, the first reference solution is used by the current single-teacher collator path.

## Outputs and Git Hygiene

Training scripts write to:

```text
runs/<run_name>/
  checkpoints/
  logs/
  eval/
  config.json
```

The following are intentionally ignored: `runs/`, `wandb/`, `slurm_logs/`, `cache/`, generated eval results, Python bytecode, checkpoints, and large generated data artifacts.

## Citation

If you use this code, please cite the beta-OPSD paper. A BibTeX entry will be added here when the public citation metadata is finalized.
