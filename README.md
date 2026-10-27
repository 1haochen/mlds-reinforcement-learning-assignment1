# MLDS Reinforcement Learning - Assignment 1

Reproducible implementations and evidence for shuttle-dispatch dynamic programming and REINFORCE on `CartPole-v1` and `Pong-v5`, with and without a learned value baseline.

**GitHub:** [1haochen/mlds-reinforcement-learning-assignment1](https://github.com/1haochen/mlds-reinforcement-learning-assignment1)  
**Submission report:** [`output/pdf/assignment1_report.pdf`](output/pdf/assignment1_report.pdf)

## What is included

- 500-stage backward induction (enumeration), infinite-horizon value iteration, and policy iteration for the shuttle MDP.
- Basic episodic REINFORCE in PyTorch for CartPole and Pong.
- A learned state-value baseline variant for both environments.
- Exact Pong preprocessing from the assignment and an action map restricted to RIGHT/LEFT (ALE actions 2/3).
- CSV/JSON evidence, labeled plots, model checkpoints, automated correctness tests, and a generated PDF report.
- Full experiment commands and smaller smoke workflows.

The checked-in CartPole evidence uses 500 training episodes and the required 500 complete evaluation episodes. Pong artifacts are clearly labeled smoke evidence (5 training, 20 truncated evaluation episodes) because full Atari training is a long-running compute job; the exact full command is provided below. No unrun result is presented as completed.

## Setup

Python 3.10+ is required. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,atari]'
```

Gymnasium exposes the assignment's namespaced Atari environment as `ALE/Pong-v5`; the CLI accepts the requested `Pong-v5` and maps it to that canonical registration. Recent `ale-py` wheels include the supported ROM installation. Confirm availability with:

```bash
python -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); gym.make('ALE/Pong-v5'); print('Pong ready')"
```

## Run the experiments

From the repository root:

```bash
# Shuttle: all three algorithms, validation statistics, CSV, and plots
python -m mlds_rl.cli --output artifacts shuttle

# CartPole Part 1 and Part 2; evaluation defaults to the required 500 episodes
python -m mlds_rl.cli --output artifacts cartpole --episodes 1500 --seed 42
python -m mlds_rl.cli --output artifacts cartpole --baseline --episodes 1500 --seed 43

# Pong Part 1 and Part 2; full runs (long-running)
python -m mlds_rl.cli --output artifacts pong --episodes 10000 --seed 42
python -m mlds_rl.cli --output artifacts pong --baseline --episodes 10000 --seed 43

# Quick end-to-end validation excluding Atari
python -m mlds_rl.cli --output artifacts-smoke all --smoke

# Rebuild the report from the recorded artifacts
python -m mlds_rl.report --artifacts artifacts --output output/pdf/assignment1_report.pdf
```

`scripts/run_all.sh` contains the full sequence. Depending on hardware, Pong can require hours. Use a single GPU if adding GPU execution, as required by the assignment.

## Validate correctness

```bash
PYTHONPATH=src pytest -q
```

The checks go beyond file formats:

- Every state-action transition row sums to one, including the 200-person cap.
- Value iteration has a small Bellman residual.
- Value iteration and exact policy iteration independently produce the same value/policy.
- The 500-stage initial policy is compared with the infinite-horizon policy.
- Deterministic arrivals test sensitivity to stochasticity.
- Discounted-return calculations are checked against a hand calculation.
- Constant returns cannot produce NaNs during normalization.
- Pong cropping, downsampling, binarization, and action restriction are checked directly.
- Full CartPole evaluation records all 500 rollouts, mean, standard deviation, and training curves.

Raw evidence is in `artifacts/**/training_rewards.csv`, `evaluation_rewards.csv`, and `summary.json`, making every reported number independently recomputable.

## Repository map

```text
src/mlds_rl/shuttle.py          Shuttle model and three DP solvers
src/mlds_rl/policy_gradient.py  Networks, preprocessing, training, evaluation
src/mlds_rl/cli.py              Reproducible experiment/plot runner
src/mlds_rl/report.py           PDF report generator
tests/                          Behavioral and numerical validation
artifacts/                      Raw data, summaries, figures, checkpoints
output/pdf/                     Canvas-ready report
docs/superpowers/               Design and implementation plan
```

## Key assumptions and edge cases

For the shuttle model, dispatch occurs first, then arrivals, the queue is capped at 200, and holding cost is charged on the resulting queue. Terminal cost at stage 500 is zero. At the boundary, excess arrivals remain represented by state 200 rather than disappearing into an invalid state.

For policy gradients, both Gymnasium termination flags are honored; each reset uses a distinct deterministic seed; one-step and constant-return episodes normalize safely; gradients are clipped; evaluation uses a greedy policy; and Pong uses frame differences so motion is observable. The report discusses additional edge cases and quantitative evidence in detail.

## Reproducibility note

PyTorch and emulator results can vary slightly across operating systems and hardware despite fixed seeds. The repository preserves configs, raw rewards, checkpoints, and package constraints so discrepancies can be diagnosed rather than hidden.

## License

MIT. See [`LICENSE`](LICENSE).

