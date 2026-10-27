"""Command-line experiment runner that writes reproducible evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .policy_gradient import TrainConfig, evaluate, save_checkpoint, train
from .shuttle import ShuttleMDP

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def _write_rows(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_shuttle(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    mdp = ShuttleMDP()
    finite_values, finite_policy = mdp.backward_induction(500)
    vi_value, vi_policy, vi_iterations = mdp.value_iteration()
    pi_value, pi_policy, pi_iterations = mdp.policy_iteration()
    deterministic = ShuttleMDP(arrivals=(3,))
    _, deterministic_policy, _ = deterministic.value_iteration()

    states = mdp.states
    _write_rows(
        output / "solutions.csv",
        ["waiting", "finite_t0_value", "finite_t0_action", "infinite_value", "vi_action", "pi_action"],
        zip(states, finite_values[0], finite_policy[0], vi_value, vi_policy, pi_policy),
    )

    plots = [
        ("finite_value.png", finite_values[0], "Finite horizon (T=500)", "Optimal cost-to-go at t=0"),
        ("infinite_value.png", vi_value, "Infinite-horizon value", "Optimal discounted cost"),
    ]
    for filename, values, title, ylabel in plots:
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        axis.plot(states, values, color="#1f5a94", linewidth=2)
        axis.set(title=title, xlabel="Customers waiting", ylabel=ylabel)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.1))
    axis.step(states, pi_policy, where="post", color="#a33d2d", linewidth=2)
    axis.set(
        title="Policy iteration: optimal action",
        xlabel="Customers waiting",
        ylabel="Action (0=wait, 1=dispatch)",
        yticks=[0, 1],
        ylim=(-0.15, 1.15),
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "optimal_policy.png", dpi=180)
    plt.close(fig)

    summary = {
        "assumptions": {
            "event_order": "dispatch, arrivals, capped next state, then holding cost",
            "terminal_cost": 0,
            "arrival_distribution": "uniform over 1,2,3,4,5",
        },
        "finite_t0_threshold": mdp.dispatch_threshold(finite_policy[0]),
        "value_iteration_threshold": mdp.dispatch_threshold(vi_policy),
        "policy_iteration_threshold": mdp.dispatch_threshold(pi_policy),
        "deterministic_arrival_threshold": deterministic.dispatch_threshold(deterministic_policy),
        "value_iteration_iterations": vi_iterations,
        "policy_iteration_iterations": pi_iterations,
        "bellman_residual": mdp.bellman_residual(vi_value),
        "max_vi_pi_value_difference": float(np.max(np.abs(vi_value - pi_value))),
        "vi_pi_policy_agreement": float(np.mean(vi_policy == pi_policy)),
        "finite_vi_policy_agreement": float(np.mean(finite_policy[0] == vi_policy)),
        "transition_row_sum_max_error": float(np.max(np.abs(mdp.transition.sum(axis=2) - 1))),
    }
    _save_json(output / "summary.json", summary)
    return summary


def _moving_average(values: list[float], window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if len(array) < window:
        return np.arange(1, len(array) + 1), np.cumsum(array) / np.arange(1, len(array) + 1)
    average = np.convolve(array, np.ones(window) / window, mode="valid")
    return np.arange(window, len(array) + 1), average


def run_policy_gradient(
    env_id: str,
    baseline: bool,
    output: Path,
    episodes: int,
    evaluation_episodes: int,
    seed: int,
    max_steps: int | None = None,
) -> dict:
    gamma = 0.95 if env_id == "CartPole-v1" else 0.99
    hidden_dim = 128 if env_id == "CartPole-v1" else 200
    learning_rate = 0.01 if env_id == "CartPole-v1" else 0.001
    config = TrainConfig(
        env_id=env_id,
        episodes=episodes,
        gamma=gamma,
        learning_rate=learning_rate,
        baseline=baseline,
        seed=seed,
        hidden_dim=hidden_dim,
        max_steps=max_steps,
        baseline_learning_rate=learning_rate,
    )
    output.mkdir(parents=True, exist_ok=True)
    result = train(config)
    evaluation = evaluate(
        result["policy"], env_id, episodes=evaluation_episodes, seed=seed + 50_000, max_steps=max_steps
    )
    save_checkpoint(result, output / "checkpoint.pt")
    _write_rows(output / "training_rewards.csv", ["episode", "reward"], enumerate(result["rewards"], 1))
    _write_rows(output / "evaluation_rewards.csv", ["episode", "reward"], enumerate(evaluation, 1))

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    episodes_axis = np.arange(1, len(result["rewards"]) + 1)
    axis.plot(episodes_axis, result["rewards"], alpha=0.35, linewidth=0.8, label="Episode reward")
    ma_x, ma = _moving_average(result["rewards"], 100)
    axis.plot(ma_x, ma, linewidth=2.2, label="100-episode moving average")
    axis.set(title=f"{env_id}: {'learned baseline' if baseline else 'basic REINFORCE'}", xlabel="Training episode", ylabel="Episode reward")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curve.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.hist(evaluation, bins=min(25, max(5, int(np.sqrt(len(evaluation))))), color="#1f5a94", edgecolor="white")
    axis.axvline(np.mean(evaluation), color="#a33d2d", linestyle="--", label=f"Mean={np.mean(evaluation):.2f}")
    axis.set(title=f"{env_id} evaluation ({len(evaluation)} episodes)", xlabel="Episode reward", ylabel="Count")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "evaluation_histogram.png", dpi=180)
    plt.close(fig)

    summary = {
        "env_id": env_id,
        "baseline": baseline,
        "training_episodes": episodes,
        "evaluation_episodes": evaluation_episodes,
        "evaluation_mean": float(np.mean(evaluation)),
        "evaluation_std": float(np.std(evaluation)),
        "last_100_training_mean": float(np.mean(result["rewards"][-100:])),
        "seed": seed,
        "max_steps": max_steps,
        "status": "full" if evaluation_episodes == 500 and max_steps is None else "smoke",
    }
    _save_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("shuttle", help="Run all shuttle dynamic-programming experiments")
    for name, env in (("cartpole", "CartPole-v1"), ("pong", "Pong-v5")):
        child = sub.add_parser(name, help=f"Train and evaluate {env}")
        child.add_argument("--baseline", action="store_true")
        child.add_argument("--episodes", type=int, default=1500 if name == "cartpole" else 10_000)
        child.add_argument("--evaluation-episodes", type=int, default=500)
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--max-steps", type=int)
    all_parser = sub.add_parser("all", help="Run shuttle and both CartPole variants")
    all_parser.add_argument("--smoke", action="store_true", help="Use small counts for pipeline validation")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "shuttle":
        print(json.dumps(run_shuttle(args.output / "shuttle"), indent=2))
    elif args.command in ("cartpole", "pong"):
        env = "CartPole-v1" if args.command == "cartpole" else "Pong-v5"
        variant = "baseline" if args.baseline else "basic"
        print(json.dumps(run_policy_gradient(env, args.baseline, args.output / args.command / variant, args.episodes, args.evaluation_episodes, args.seed, args.max_steps), indent=2))
    else:
        run_shuttle(args.output / "shuttle")
        episodes, evaluation = ((40, 50) if args.smoke else (1500, 500))
        for baseline in (False, True):
            variant = "baseline" if baseline else "basic"
            run_policy_gradient("CartPole-v1", baseline, args.output / "cartpole" / variant, episodes, evaluation, 42 + int(baseline))
        print(f"Artifacts written to {args.output}")


if __name__ == "__main__":
    main()
