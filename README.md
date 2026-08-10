# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

<p align="center">
<a href="https://arxiv.org/pdf/2601.18734v3"><img src="https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg"></a>
<a href="https://siyan-zhao.github.io/blog/2026/opsd/"><img src="https://img.shields.io/badge/Blog-Post-blue.svg"></a>
</p>

---

## Overview

**On-Policy Self-Distillation (OPSD)** trains a single model to act as both student and teacher by conditioning on different contexts — the student sees only the problem, while the teacher additionally sees the ground-truth solution — and performs token-level distribution matching along the student's own on-policy trajectories.

This release contains only source code, launch configs, reusable evaluation inputs, and documentation. Training outputs, checkpoints, W&B runs, Slurm logs, generated evaluation samples, caches, and generated reference files are intentionally excluded. The paper PDF is included at [`docs/beta-opsd-paper.pdf`](docs/beta-opsd-paper.pdf).

## Updates

- **Apr 20, 2026**: Added per-dataset token budget override in eval (`--dataset_max_tokens`) and routed all Slurm output to `slurm_logs/`.

- **Mar 18, 2026**: Released updated code.

  (1) Fixed chat template and ZeRO-2 bugs (see [template issue](https://github.com/huggingface/trl/issues/5241)); re-ran experiments with updated results (detailed results & ablations updated on arXiv/blog). The fixes yield improved OPSD performance, most notably on Qwen3-1.7B.

  (2) Added per-token pointwise KL clipping (`--jsd_token_clip`). Style tokens (e.g. "wait", "think") exhibit 6–15× higher KL divergence than math-related tokens and dominate the training signal; clipping stabilizes training and improves performance.

- **Mar 3, 2026**: Initial code release.

## Installation

```bash
conda env create -f environment.yml
conda activate vllm
```

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

If you encounter difficulties installing flash-attn, check the version matching your CUDA and PyTorch versions from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases).

Key versions: PyTorch 2.8.0, transformers 4.57.1, trl 0.26.0, vLLM 0.11.0.

## Repository Structure

```
├── opsd_trainer.py                      # Default OPSDTrainer used by opsd_train.py
├── beta_opsd_trainer.py                 # Beta-OPSD trainer for mix-target and mixed-sampling experiments
├── data_collator.py                     # Data collator for self-distillation
├── opsd_train.py                        # Default OPSD training entry point
├── beta_opsd_train.py                   # Beta-OPSD mix-target / mixed-sampling training entry point
├── sft_train.py                         # SFT baseline training entry point
├── grpo_train.py                        # GRPO baseline training entry point
├── accelerate.yaml                      # Accelerate/DeepSpeed ZeRO-2 config
├── scripts/
│   ├── run_opsd_1b.sh                   # Training script: Qwen3-1.7B (4 GPU)
│   ├── run_opsd_4b.sh                   # Training script: Qwen3-4B (8 GPU)
│   ├── run_opsd_8b.sh                   # Training script: Qwen3-8B (8 GPU)
│   ├── run_sft.sh                       # SFT baseline
│   ├── run_grpo.sh                      # GRPO baseline
│   ├── beta_opsd_mixed_sampling*.sbatch # Mixed-sampling launchers; target is fixed teacher
│   ├── beta_opsd_mix_target_qwen34b_h200.sbatch # Qwen3-4B mix-target launcher
│   ├── submit_opsd_1b_localvllm.sbatch  # Slurm: 1.7B training + local vLLM
│   ├── submit_opsd_4b_localvllm.sbatch  # Slurm: 4B training + local vLLM
│   ├── submit_opsd_8b_localvllm.sbatch  # Slurm: 8B training + local vLLM
│   └── submit_vllm_server.sbatch        # Slurm: standalone vLLM inference server
└── eval/
    ├── eval_math.py                     # vLLM-based evaluation on math benchmarks
    └── submit.sh                        # Slurm eval submission script
```

Output layout per run:

```
runs/<run_name>/
  checkpoints/   # HF Trainer checkpoints (LoRA adapters)
  logs/          # training stdout/stderr
  eval/          # eval_math.py JSON + JSONL outputs
  config.json    # hyperparameters captured at launch
slurm_logs/      # Slurm job output files (gitignored)
```

## Quick Start

Reproduce results on Qwen3-1.7B (training peaks within ~100 steps, ~15 min on 4×H100):

```bash
# On a Slurm cluster:
RUN=qwen31b_opsd_run1 sbatch scripts/submit_opsd_1b_localvllm.sbatch

# Or directly (e.g. from an interactive node with GPUs):
RUN=qwen31b_opsd_run1 bash scripts/run_opsd_1b.sh
```

Evaluate a checkpoint:

```bash
RUN=qwen31b_opsd_run1 STEP=100 sbatch eval/submit.sh
```

### Evaluation Results on Qwen3-1.7B (pass@12, thinking mode)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|------|--------|
| Base | 51.5%  |
| 25   | 51.4%  |
| 50   | 52.8%  |
| 75   | 54.4%  |
| 100  | 57.2%  |

</td>
<td>

| Step | Avg@12 |
|------|--------|
| Base | 36.7%  |
| 25   | 42.5%  |
| 50   | 43.9%  |
| 75   | 40.6%  |
| 100  | 41.1%  |

</td>
<td>

| Step | Avg@12 |
|------|--------|
| Base | 23.1%  |
| 25   | 24.7%  |
| 50   | 27.8%  |
| 75   | 26.9%  |
| 100  | 29.2%  |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, thinking mode enabled, max\_new\_tokens=38912, top-p=none, top-k disabled, min-p=0, presence\_penalty=0, num\_samples=12

## Training

### OPSD

See [`scripts/run_opsd_1b.sh`](scripts/run_opsd_1b.sh), [`scripts/run_opsd_4b.sh`](scripts/run_opsd_4b.sh), [`scripts/run_opsd_8b.sh`](scripts/run_opsd_8b.sh).

#### Key OPSD arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fixed_teacher` | `False` | Freeze the teacher at the initial checkpoint (step 0). Requires `--use_peft`. Without PEFT the teacher updates every step, which can be unstable. |
| `--use_ema_teacher` | `False` | Use an exponential moving average of student weights as the teacher instead of a frozen copy. |
| `--beta` | `0` | JSD mixture weight. `0` = forward KL (student → teacher), `1` = reverse KL. |
| `--temperature` | `1.1` | Softmax temperature applied to teacher logits. |
| `--jsd_token_clip` | `0.05` | Per-token KL clip threshold. Prevents style tokens ("wait", "think") from dominating the gradient. |
| `--max_completion_length` | — | Student on-policy generation budget (tokens). We use 1024 in main experiments. |
| `--use_tinker_loss` | `False` | Sampled-token reverse-KL policy-gradient loss (memory-efficient; no clipping yet). |
| `--reason_first` | `False` | Prepend an explicit teacher rationalization to the teacher context before distillation. |
| `--run_config` | `None` | Run name used for the output directory and WandB. |

### Beta-OPSD

Use [`beta_opsd_train.py`](beta_opsd_train.py) and [`beta_opsd_trainer.py`](beta_opsd_trainer.py) for the beta-OPSD experiments.

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_tinker_loss` | `True` | Default beta-OPSD sampled-token loss. |
| `--use_mixed_teacher_target` | `True` | Mix the reference student target with privileged teacher logits. |
| `--mixed_teacher_target_teacher_weight` | `0.5` | Weight of teacher logits in the target interpolation. |
| `--mixed_teacher_target_teacher_weight_linear_decay` | `False` | Linearly schedule the target teacher weight. |
| `--mixed_teacher_target_teacher_weight_final` | `0.5` | Final teacher weight for the linear target schedule. |
| `--use_mixed_sampling` | `False` | Sample from a student/teacher proposal mixture. When enabled, the loss target is forced to the fixed teacher. |
| `--disable_mixed_teacher_target` | `False` | Disable the default mixed teacher target for fixed-teacher baselines. |

#### vLLM integration

OPSD uses vLLM for fast on-policy student generation. Two modes:

| Mode | Flag | Description |
|------|------|-------------|
| Colocated (default) | `--vllm_mode colocate` | vLLM shares GPUs with training (uses `--vllm_gpu_memory_utilization` to split VRAM). |
| Local server | `--vllm_mode server` | vLLM runs on a dedicated GPU on the same node. Use `submit_opsd_*_localvllm.sbatch`. |
| Remote server | `--vllm_mode server` + `VLLM_SERVER_HOST=<node>` | vLLM runs on a separate Slurm node. Launch `submit_vllm_server.sbatch` first. |

### SFT Baseline

See [`scripts/run_sft.sh`](scripts/run_sft.sh).

### GRPO Baseline

See [`scripts/run_grpo.sh`](scripts/run_grpo.sh).

## Evaluation

`eval/eval_math.py` evaluates checkpoints on AIME24, AIME25, and HMMT25 using vLLM.

```bash
# Eval a LoRA checkpoint
RUN=qwen31b_opsd_run1 STEP=100 sbatch eval/submit.sh

# Eval the base model only
BASE_MODEL=Qwen/Qwen3-1.7B sbatch eval/submit.sh
```

Key eval arguments:

| Argument | Description |
|----------|-------------|
| `--thinking` | Enable thinking mode (Qwen3 style). |
| `--decoding greedy\|sampling` | Decoding strategy. |
| `--k N` | Number of samples per problem (pass@k). |
| `--dataset_max_tokens ds:N,...` | Per-dataset token budget override, e.g. `aime24:32768,aime25:32768`. Falls back to `--max_new_tokens`. |

## Acknowledgements

Our implementation builds on [TRL GOLD Trainer](https://huggingface.co/docs/trl/gold_trainer). We sincerely thank [@simran135](https://github.com/simran135) and [@beanie00](https://github.com/beanie00) for identifying the prompt template bugs and the ZeRO-2 issue, respectively!

## Citation

If you find this useful, please consider citing:

```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```
