import numpy as np
import pytest
import torch

from muzero.config import MuZeroConfig
from muzero.mcts import MCTS
from muzero.models import MuZeroNetwork


def my_test_config():
    config = MuZeroConfig()
    config.num_simulations = 30
    return config


def my_test_model():
    config = my_test_config()
    # Fix random generator seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = MuZeroNetwork(config)
    model = torch.compile(model)
    return model


def test_mcts():
    config = my_test_config()
    model = my_test_model()
    model.eval()
    workers = 1
    observations = np.random.random((workers, 19, 10, 9))
    root, extra_info = MCTS(config).run(model, observations, [1, 2, 3], 1, True)
    expected_shape = (workers, config.channels, 10, 9)
    assert root.hidden_state.shape == expected_shape
