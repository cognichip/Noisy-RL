# Rate or Fate? RLVeR: Reinforcement Learning with Verifiable Noisy Rewards

This repository contains the training code, paper, and project page for our research on RLVR under noisy reward signals.

**Project Page:**  https://cognichip.github.io/Noisy-RL/

## The Question

Does noisy reward change the *rate* of learning or its *fate*?

## Key Finding

We discover a sharp phase transition in RLVR governed by Youden's index:

```
J = TPR - FPR = (1 - FN) - FP
```

| Condition | Outcome |
|-----------|---------|
| J > 0 | Learning proceeds, bad modes decay to zero |
| J = 0 | Neutral drift, no net learning |
| J < 0 | Anti-learning, model collapses to wrong answers |

When J > 0, noise affects the rate of learning, not its fate.

## Repository Structure

```
Noisy-RL/
├── docs/                           # Project website (GitHub Pages)
├── data/
│   ├── training/                   # Training split
│   └── validation/                 # Validation split
└── code/
    └── training/
        ├── verl/recipe/RLVeR       # GRPO implementation with noisy rewards
        └── README.md               # Training instructions
```

## Data

We use a filtered subset of 10k high-quality samples from the Open-R1 project:

**Source:** [open-r1/verifiable-coding-problems-python](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python)



## Quick Start

```bash
git clone https://github.com/cognichip/Noisy-RL.git
cd Noisy-RL/code/training/verl
pip install -e .
```

See [code/training/README.md](code/training/README.md) for detailed training instructions.

## Running Experiments

Edit paths in `code/training/verl/recipe/RLVeR/run_RLVeR.sh`:

```bash
MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN="path/to/train.parquet"
VAL="path/to/val.parquet"

FALSE_POSITIVE_RATE=0.1
FALSE_NEGATIVE_RATE=0.1
```

Then run:

```bash
cd code/training/verl/recipe/RLVeR
bash run_RLVeR.sh
```

## Abstract

RLVR is simple but powerful: sample an answer, verify it, and update the model. But in practice the verifier is almost never clean—unit tests probe only finitely many corner cases, human/synthetic labels are imperfect, and LLM judges can be biased and prone to reward hacking.

We leverage a novel analytical framework to study the evolution of algorithms like GRPO under general noise levels. Modeling each prompt as a multi-armed bandit over recurring reasoning modes, we derive a tractable probability-simplex flow with a sharp noise threshold. The dynamics decouple into inner competition among correct modes and an outer mean-field ODE for the total bad-mode mass, whose drift depends only on Youden's index J = TPR - FPR.

## Citation

```bibtex
@article{rlver2025,
    title={Rate or Fate? RLVeR: Reinforcement Learning 
           with Verifiable Noisy Rewards},
    author={Anonymous},
    journal={arXiv preprint},
    year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
