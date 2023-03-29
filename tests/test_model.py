import numpy as np
import pytest
import torch
from gymxq.constants import NUM_ACTIONS

from muzero.config import NUM_COL, NUM_ROW, PLANE_NUM, MuZeroConfig
from muzero.feature_utils import encoded_action
from muzero.models import MuZeroNetwork


def my_test_config():
    config = MuZeroConfig()
    # = plane number
    config.channels = 17
    config.reduced_channels_reward = 19
    config.reduced_channels_value = 19
    config.reduced_channels_policy = 19
    config.resnet_fc_reward_layers = [19, 19]
    config.resnet_fc_value_layers = [19, 19]
    config.resnet_fc_policy_layers = [19, 19]
    return config


def my_test_model():
    config = my_test_config()
    # Fix random generator seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = MuZeroNetwork(config)
    model = torch.compile(model)
    return model


# def test_representation():
#     workers = 8
#     model = my_test_model()
#     state_shape = (workers, PLANE_NUM, NUM_ROW, NUM_COL)
#     observations = torch.rand(state_shape, dtype=torch.float32)
#     encoded_state = model.representation(observations)
#     assert encoded_state.shape == state_shape


# def test_dynamics():
#     workers = 8
#     model = my_test_model()
#     encoded_action_shape = (workers, 2, NUM_ROW, NUM_COL)
#     state_shape = (workers, PLANE_NUM, NUM_ROW, NUM_COL)
#     observations = torch.rand(state_shape, dtype=torch.float32)
#     actions = torch.rand(encoded_action_shape, dtype=torch.float32)
#     encoded_state, reward = model.dynamics(observations, actions)
#     assert encoded_state.shape == state_shape
#     assert reward.shape == (workers,)


def test_inference():
    workers = 8
    model = my_test_model()
    encoded_action_shape = (workers, 2, NUM_ROW, NUM_COL)
    state_shape = (workers, PLANE_NUM, NUM_ROW, NUM_COL)
    actions = torch.rand(encoded_action_shape, dtype=torch.float32)
    observations = torch.rand(state_shape, dtype=torch.float32)
    (
        root_predicted_value,
        reward,
        policy_logits,
        hidden_state,
    ) = model.inference(observations, actions)
    assert root_predicted_value.shape == (workers,)
    assert reward.shape == (workers,)
    assert policy_logits.shape == (workers, NUM_ACTIONS)
    assert hidden_state.shape == state_shape
