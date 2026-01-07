# Analysis Notebooks

Interactive notebooks demonstrating the key concepts from the paper.

## Notebooks

| Notebook | Description |
|:---------|:------------|
| [01_phase_transition_simulation.ipynb](01_phase_transition_simulation.ipynb) | Simulates the phase transition governed by Youden's index J |
| [02_simplex_flow.ipynb](02_simplex_flow.ipynb) | Visualizes probability flow on the simplex for different J values |
| [03_stochastic_dynamics.ipynb](03_stochastic_dynamics.ipynb) | Demonstrates Wright-Fisher diffusion and KL regularization effects |

## Quick Start

```bash
pip install numpy matplotlib
jupyter notebook
```

## Key Concepts

### Phase Transition (Notebook 01)
Shows how the bad mode probability $p(t)$ evolves under different noise levels:
- **J > 0**: $p(t) \to 0$ (learning succeeds)
- **J = 0**: Neutral drift (no learning)
- **J < 0**: $p(t) \to 1$ (anti-learning)

### Simplex Flow (Notebook 02)
Visualizes trajectories on the probability simplex (2 good modes + 1 bad mode), showing how probability mass flows under noisy rewards.

### Stochastic Dynamics (Notebook 03)
Demonstrates the Wright-Fisher diffusion that governs competition among modes, and how KL regularization maintains diversity.
