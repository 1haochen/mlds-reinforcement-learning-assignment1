"""Tabular dynamic programming for the shuttle-dispatch MDP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShuttleMDP:
    """Finite-state discounted-cost shuttle dispatch problem.

    A decision is made from the current queue, dispatch removes up to ``capacity``,
    arrivals occur, and holding cost is charged on the resulting capped queue.
    """

    capacity: int = 15
    dispatch_cost: float = 100.0
    holding_cost: float = 2.0
    arrivals: tuple[int, ...] = (1, 2, 3, 4, 5)
    max_waiting: int = 200
    gamma: float = 0.95

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.max_waiting <= 0:
            raise ValueError("capacity and max_waiting must be positive")
        if not self.arrivals or any(a < 0 for a in self.arrivals):
            raise ValueError("arrivals must be a non-empty sequence of nonnegative integers")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        self.states = np.arange(self.max_waiting + 1)
        self.actions = np.array([0, 1], dtype=int)
        self.transition, self.cost = self._build_model()

    def _build_model(self) -> tuple[np.ndarray, np.ndarray]:
        n_states = self.max_waiting + 1
        transition = np.zeros((n_states, 2, n_states), dtype=np.float64)
        cost = np.zeros((n_states, 2), dtype=np.float64)
        probability = 1.0 / len(self.arrivals)
        for state in self.states:
            for action in self.actions:
                after_dispatch = max(0, state - self.capacity * action)
                for arrival in self.arrivals:
                    next_state = min(self.max_waiting, after_dispatch + arrival)
                    transition[state, action, next_state] += probability
                    immediate = self.dispatch_cost * action + self.holding_cost * next_state
                    cost[state, action] += probability * immediate
        return transition, cost

    def q_values(self, value: np.ndarray) -> np.ndarray:
        return self.cost + self.gamma * np.einsum("sak,k->sa", self.transition, value)

    def backward_induction(self, horizon: int = 500) -> tuple[np.ndarray, np.ndarray]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        values = np.zeros((horizon + 1, len(self.states)), dtype=np.float64)
        policy = np.zeros((horizon, len(self.states)), dtype=np.int8)
        for time in range(horizon - 1, -1, -1):
            q = self.cost + self.gamma * np.einsum(
                "sak,k->sa", self.transition, values[time + 1]
            )
            policy[time] = np.argmin(q, axis=1)
            values[time] = np.min(q, axis=1)
        return values, policy

    def value_iteration(
        self, tolerance: float = 1e-10, max_iterations: int = 100_000
    ) -> tuple[np.ndarray, np.ndarray, int]:
        value = np.zeros(len(self.states), dtype=np.float64)
        for iteration in range(1, max_iterations + 1):
            q = self.q_values(value)
            updated = np.min(q, axis=1)
            if np.max(np.abs(updated - value)) <= tolerance:
                value = updated
                return value, np.argmin(self.q_values(value), axis=1), iteration
            value = updated
        raise RuntimeError("value iteration did not converge")

    def evaluate_policy(self, policy: np.ndarray) -> np.ndarray:
        if policy.shape != (len(self.states),) or not np.isin(policy, self.actions).all():
            raise ValueError("policy must contain one valid action per state")
        rows = np.arange(len(self.states))
        p_policy = self.transition[rows, policy]
        c_policy = self.cost[rows, policy]
        return np.linalg.solve(np.eye(len(self.states)) - self.gamma * p_policy, c_policy)

    def policy_iteration(
        self, tolerance: float = 1e-10, max_iterations: int = 1_000
    ) -> tuple[np.ndarray, np.ndarray, int]:
        policy = np.zeros(len(self.states), dtype=np.int8)
        for iteration in range(1, max_iterations + 1):
            value = self.evaluate_policy(policy)
            improved = np.argmin(self.q_values(value), axis=1).astype(np.int8)
            if np.array_equal(improved, policy):
                if self.bellman_residual(value) > max(tolerance, 1e-8):
                    raise RuntimeError("policy stabilized without satisfying Bellman equation")
                return value, policy, iteration
            policy = improved
        raise RuntimeError("policy iteration did not converge")

    def bellman_residual(self, value: np.ndarray) -> float:
        return float(np.max(np.abs(value - np.min(self.q_values(value), axis=1))))

    @staticmethod
    def dispatch_threshold(policy: np.ndarray) -> int | None:
        indices = np.flatnonzero(policy == 1)
        return int(indices[0]) if len(indices) else None

