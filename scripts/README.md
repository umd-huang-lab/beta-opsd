# Scripts

The public scripts are organized around reusable methods, not one-off cluster jobs.

## Training Entry Points

Run a paper recipe directly:

```bash
CONFIG=recipes/paper/beta_opsd_mix_target_qwen3_4b.yaml bash scripts/train/beta_opsd.sh
CONFIG=recipes/paper/beta_opsd_mixed_sampling_qwen3_1_7b.yaml bash scripts/train/beta_opsd.sh
CONFIG=recipes/paper/opsd_fixed_teacher_qwen3_1_7b.yaml bash scripts/train/opsd.sh
CONFIG=recipes/paper/sft_qwen3_4b.yaml bash scripts/train/sft.sh
CONFIG=recipes/paper/grpo_qwen3_4b.yaml bash scripts/train/grpo.sh
```

Each launcher calls `scripts/train/run_recipe.py`, which converts the YAML recipe into an `accelerate launch` command and writes `runs/<RUN>/config.json`.

Common overrides:

```bash
RUN=my_run MAX_STEPS=10 MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B \
  CONFIG=recipes/paper/beta_opsd_mix_target_qwen3_4b.yaml \
  bash scripts/train/beta_opsd.sh
```

## Slurm Templates

The files in `scripts/slurm/` are portable templates. They intentionally do not encode local accounts, partitions, node names, or GPU SKUs.

```bash
sbatch --account=<acct> --partition=<part> --gres=gpu:1 scripts/slurm/submit_beta_opsd.sbatch
sbatch --export=ALL,CONFIG=recipes/paper/sft_qwen3_1_7b.yaml --gres=gpu:4 scripts/slurm/submit_sft.sbatch
```

## Smoke Tests

Smoke tests use the same recipe runner with short training settings:

```bash
bash scripts/smoke/beta_opsd_mix_target.sh
bash scripts/smoke/beta_opsd_mixed_sampling.sh
bash scripts/smoke/opsd.sh
bash scripts/smoke/sft.sh
bash scripts/smoke/grpo.sh
```

## vLLM Helpers

`scripts/vllm/serve.sh` starts a TRL vLLM server. `scripts/vllm/run_with_local_server.sh` can run a training launcher while reserving separate GPUs for a local server.
