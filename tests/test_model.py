import numpy as np
import pytest
import torch
from gymxq.constants import NUM_ACTIONS

from muzero.config import NUM_COL, NUM_ROW, STATE_PLANE_NUM, MuZeroConfig
from muzero.models import MuZeroNetwork


def my_test_config():
    config = MuZeroConfig()
    return config


def my_test_model():
    config = my_test_config()
    # Fix random generator seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = MuZeroNetwork(config)
    model = torch.compile(model)
    return model


def test_dynamics():
    config = my_test_config()
    workers = 2
    model = my_test_model()
    encoded_action_shape = (workers, 2, NUM_ROW, NUM_COL)
    state_shape = (workers, STATE_PLANE_NUM, NUM_ROW, NUM_COL)
    observations = torch.rand(state_shape, dtype=torch.float32)
    actions = torch.rand(encoded_action_shape, dtype=torch.float32)
    encoded_state = model.representation(observations)
    next_state, reward = model.dynamics(encoded_state, actions)
    expected_shape = (workers, config.channels, NUM_ROW, NUM_COL)
    assert next_state.shape == expected_shape
    assert reward.shape == (workers,)


def test_inference():
    config = my_test_config()
    workers = 6
    model = my_test_model()
    encoded_action_shape = (workers, 2, NUM_ROW, NUM_COL)
    state_shape = (workers, STATE_PLANE_NUM, NUM_ROW, NUM_COL)
    actions = torch.rand(encoded_action_shape, dtype=torch.float32)
    observations = torch.rand(state_shape, dtype=torch.float32)
    (
        root_predicted_value,
        reward,
        policy_logits,
        next_state,
    ) = model.inference(observations, actions)

    expected_shape = (workers, config.channels, NUM_ROW, NUM_COL)
    assert root_predicted_value.shape == (workers,)
    assert reward.shape == (workers,)
    assert policy_logits.shape == (workers, NUM_ACTIONS)
    assert next_state.shape == expected_shape

    # 递归推理
    actions = torch.rand(encoded_action_shape, dtype=torch.float32)
    (
        root_predicted_value,
        reward,
        policy_logits,
        next_state,
    ) = model.inference(next_state, actions)
    assert root_predicted_value.shape == (workers,)
    assert reward.shape == (workers,)
    assert policy_logits.shape == (workers, NUM_ACTIONS)
    assert next_state.shape == expected_shape
