# RLVeR Training

GRPO training with noisy reward verification, built on the verl framework.

## Data

We use a filtered subset of 10k high-quality samples from the Open-R1 project's verifiable coding problems dataset:

**Source:** [open-r1/verifiable-coding-problems-python](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python)

The samples were selected based on problem clarity, test case coverage, and solution verifiability. We provide the preprocessed train/test splits in the `/data` directory:

```
data/
├── validation/open-r1-python/train.parquet    
└── training/open-r1-pythontest.parquet     
```

Each sample contains:
- `prompt`: the coding problem description
- `verification_info`: dictionary with `test_cases` for sandbox evaluation

## Prerequisites

1. Install verl and dependencies:
```bash
cd verl
pip install -e .
```

2. Get an E2B API key from https://e2b.dev and set it in `reward_sandbox.py`

3. Prepare your training data in parquet format with columns:
   - `prompt`: the coding problem
   - `verification_info`: dict with `test_cases` list

## Quick Start

Edit the paths in `run_RLVeR.sh`:

```bash
MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN="path/to/train.parquet"
VAL="path/to/val.parquet"
REWARD="$(pwd)/verl/recipe/RLVeR/batch_reward_python_sandbox_with_noise.py"
```

Set your noise levels:

```bash
FALSE_POSITIVE_RATE=0.1   # probability of rewarding wrong solutions
FALSE_NEGATIVE_RATE=0.1   # probability of penalizing correct solutions
```

Run:

```bash
cd verl/recipe/RLVeR
bash run_RLVeR.sh
```

## Noise Parameters

The key insight from the paper: learning depends on Youden's index J = (1 - FN) - FP.

| Setting | FP | FN | J | Outcome |
|---------|----|----|---|---------|
| Clean verifier | 0.0 | 0.0 | 1.0 | Fast learning |
| Moderate noise | 0.1 | 0.1 | 0.8 | Slower but stable |
| Critical threshold | 0.5 | 0.5 | 0.0 | No learning |
| Adversarial | 0.6 | 0.5 | -0.1 | Anti-learning |

When J > 0, noise affects the rate of learning, not its fate.
When J < 0, the model learns to produce wrong answers.

## Files

- `run_RLVeR.sh` - main training script with all hyperparameters
- `batch_reward_python_sandbox_with_noise.py` - reward function with noise injection
- `batch_reward_python_sandbox.py` - clean reward function (no noise)
- `reward_sandbox.py` - core sandbox execution using E2B

## Key Hyperparameters

From `run_RLVeR.sh`:

```bash
data.train_batch_size=16
data.max_prompt_length=4000
data.max_response_length=4000
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.rollout.n=8        # samples per prompt
trainer.total_epochs=10
```

## Logging

Training logs to wandb under project `multi-arm-bandit`. Experiment names follow the pattern:
```
noise_B{youden}_FP{fp_rate}_FN{fn_rate}
```

Logs also saved locally to `verl_demo_YYYYMMDD_HHMMSS.log`.

## Notes

- Noise is only applied to the training split; validation uses clean rewards
- The sandbox evaluates code by running test cases in an E2B container
- Binary rewards: 1.0 if all tests pass, 0.0 otherwise
- Percentage rewards available via `sandbox_percentage` mode
