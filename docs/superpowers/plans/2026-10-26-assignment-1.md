# Assignment 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, document, and publish the complete reproducible Assignment 1 repository and PDF report.

**Architecture:** A focused Python package exposes tabular dynamic-programming and PyTorch REINFORCE APIs. Thin CLI and report layers persist reproducible artifacts and turn verified results into the submission PDF.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, Gymnasium, ALE/Pong, Matplotlib, ReportLab, pytest.

---

### Task 1: Project contract and shuttle tests

**Files:** Create `pyproject.toml`, `requirements.txt`, `tests/test_shuttle.py`, `tests/test_policy_gradient.py`.

- [ ] Write tests for normalized shuttle transitions, boundary capping, finite-horizon shapes, Bellman residual, policy-iteration agreement, discounted returns, Pong preprocessing, and action mapping.
- [ ] Run the tests and confirm they fail because package modules do not exist.
- [ ] Commit the red tests.

### Task 2: Shuttle dynamic programming

**Files:** Create `src/mlds_rl/__init__.py`, `src/mlds_rl/shuttle.py`.

- [ ] Implement the explicit transition model and immediate expected costs.
- [ ] Implement 500-stage backward induction, value iteration, and policy iteration.
- [ ] Expose Bellman residual and policy evaluation helpers.
- [ ] Run shuttle tests and confirm they pass.

### Task 3: Policy gradient

**Files:** Create `src/mlds_rl/policy_gradient.py`.

- [ ] Implement discounted returns and stable normalization.
- [ ] Implement CartPole and Pong policy networks plus Pong preprocessing and action mapping.
- [ ] Implement seeded REINFORCE training with optional learned value baseline.
- [ ] Implement deterministic/stochastic evaluation and checkpoint persistence.
- [ ] Run policy-gradient tests and confirm they pass.

### Task 4: Experiment and plotting CLI

**Files:** Create `src/mlds_rl/cli.py`, `scripts/run_all.sh`.

- [ ] Add `shuttle`, `cartpole`, `pong`, and `all` commands with full and smoke settings.
- [ ] Write CSV/JSON data and labeled PNG plots for every run.
- [ ] Include random-policy validation and exact 500-episode evaluation defaults.
- [ ] Run shuttle and CartPole experiments and inspect artifacts.

### Task 5: Documentation and PDF report

**Files:** Create `README.md`, `src/mlds_rl/report.py`, `.gitignore`, `LICENSE`.

- [ ] Document setup, exact commands, artifact layout, assumptions, edge cases, validation rationale, and limitations.
- [ ] Generate the PDF report from experiment artifacts with repository metadata.
- [ ] Render every PDF page and inspect for clipping, overlap, or broken glyphs.

### Task 6: Final verification and publication

**Files:** All repository files and generated artifacts.

- [ ] Run the complete pytest suite.
- [ ] Run CLI help and smoke workflows.
- [ ] Check required artifact files and report text.
- [ ] Initialize Git, commit, create the GitHub repository, push, and verify the remote URL.

