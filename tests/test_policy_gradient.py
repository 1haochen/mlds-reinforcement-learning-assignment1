import numpy as np
import torch

from mlds_rl.policy_gradient import (
    action_index_to_pong,
    discounted_returns,
    normalize,
    preprocess_pong,
)


def test_discounted_returns_matches_hand_calculation():
    actual = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
    assert torch.allclose(actual, torch.tensor([2.75, 3.5, 3.0]))


def test_normalize_constant_vector_is_finite_and_zero():
    values = torch.tensor([4.0, 4.0, 4.0])
    actual = normalize(values)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, torch.zeros(3))


def test_pong_preprocessing_shape_values_and_background():
    image = np.full((210, 160, 3), 144, dtype=np.uint8)
    image[40, 20, 0] = 255
    actual = preprocess_pong(image)
    assert actual.shape == (80, 80)
    assert actual.dtype == np.float32
    assert set(np.unique(actual)).issubset({0.0, 1.0})
    assert actual.sum() == 1


def test_pong_action_mapping_restricts_agent_to_left_and_right():
    assert action_index_to_pong(0) == 2
    assert action_index_to_pong(1) == 3

