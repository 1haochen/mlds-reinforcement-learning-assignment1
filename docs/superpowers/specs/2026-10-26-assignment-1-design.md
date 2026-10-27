# Assignment 1 Repository Design

## Goal

Produce a reproducible GitHub repository and PDF report covering every requested Assignment 1 component: shuttle-dispatch dynamic programming; REINFORCE on CartPole-v1 and Pong-v5; a baseline variant; plots; edge cases; quantitative validation; run instructions; and a three-month extension plan.

## Architecture

The `mlds_rl` package separates reusable algorithms from command-line experiment drivers. Shuttle transition/cost construction and three solvers live in one module. Policy-gradient networks, preprocessing, return calculation, training, evaluation, and checkpoint I/O live in a second module. A single CLI runs experiments and writes CSV/JSON/PNG artifacts under `artifacts/`. A report generator reads those artifacts and creates `output/pdf/assignment1_report.pdf`.

## Mathematical conventions

- Shuttle state is the number waiting at the decision epoch, `0..200`.
- Action 0 waits; action 1 dispatches one capacity-15 shuttle.
- Dispatch first removes up to 15 customers, arrivals then occur, and the state is capped at 200.
- Per-step cost is dispatch cost plus holding cost on customers left after dispatch and arrival.
- Finite-horizon enumeration uses 500 stages and zero terminal cost.
- Infinite-horizon value and policy iteration use discount 0.95 and tight numerical tolerances.
- REINFORCE optimizes discounted episodic return using Adam. CartPole uses gamma 0.95; Pong uses gamma 0.99 and only actions 2/3.
- The baseline variant uses a learned state-value network and advantage `return - value`.

## Evidence and outputs

Each experiment saves raw episode rewards, evaluation rewards, summary statistics, plots, configuration, seed, and checkpoints. Required training plots show raw reward and a 100-episode moving average. Required evaluation plots are histograms over 500 episodes for full runs. Smoke runs use smaller counts and are labeled as smoke evidence.

Validation includes transition-probability checks, finite/infinite-policy agreement, Bellman residuals, deterministic-arrival sensitivity, return-calculation checks, Pong image-shape/background checks, action restriction checks, seed control, NaN-safe reward normalization, and learning comparisons against a random policy.

## Error handling and reproducibility

Commands fail with actionable messages when optional Atari dependencies or ROMs are unavailable. Output directories are created safely. Seeds cover Python, NumPy, PyTorch, environment resets, and action spaces. Checkpoints include configuration and model parameters.

## Scope

The repository provides full experiment commands and generated evidence available in the current environment. It does not invent long-running Pong results when the required emulator or compute run is unavailable.

