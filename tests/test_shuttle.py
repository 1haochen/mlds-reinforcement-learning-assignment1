import numpy as np

from mlds_rl.shuttle import ShuttleMDP


def test_transition_rows_are_probabilities_and_respect_cap():
    mdp = ShuttleMDP(max_waiting=200)
    assert np.allclose(mdp.transition.sum(axis=2), 1.0)
    assert np.all(mdp.transition >= 0)
    assert mdp.transition[200, 0, 200] == 1.0


def test_dispatch_removes_capacity_before_arrivals():
    mdp = ShuttleMDP(capacity=15, arrivals=(1,))
    next_state = np.argmax(mdp.transition[20, 1])
    assert next_state == 6
    assert mdp.cost[20, 1] == 100 + 2 * 6


def test_finite_horizon_and_infinite_solvers_agree_near_start():
    mdp = ShuttleMDP()
    finite_value, finite_policy = mdp.backward_induction(horizon=500)
    value, vi_policy, iterations = mdp.value_iteration(tolerance=1e-10)
    pi_value, pi_policy, _ = mdp.policy_iteration(tolerance=1e-10)
    assert finite_value.shape == (501, 201)
    assert finite_policy.shape == (500, 201)
    assert iterations < 10_000
    assert mdp.bellman_residual(value) < 1e-8
    assert np.max(np.abs(value - pi_value)) < 1e-6
    assert np.array_equal(vi_policy, pi_policy)
    assert np.mean(finite_policy[0] == vi_policy) > 0.98


def test_deterministic_arrival_model_is_valid():
    mdp = ShuttleMDP(arrivals=(3,))
    assert np.allclose(mdp.transition.sum(axis=2), 1.0)
    _, policy, _ = mdp.value_iteration()
    assert set(np.unique(policy)).issubset({0, 1})

