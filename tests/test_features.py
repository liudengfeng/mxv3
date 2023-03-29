import gymnasium as gym
import gymxq
import pytest
import xqcpp
from muzero.feature_utils import obs2feature
from muzero.config import MuZeroConfig


def test_env_features():
    init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 192"
    config = MuZeroConfig()
    environment = gym.make("xqv1", init_fen=init_fen, render_mode="ansi")
    obs, info = environment.reset()
    observation = obs2feature(obs, info, flatten=False)
    assert observation.shape == config.encoded_observation_shape
    action = xqcpp.m2a("2838")
    obs, reward, termination, truncation, info = environment.step(action)
    assert reward == 1
    assert termination
    assert not truncation
    observation = obs2feature(obs, info, flatten=False)
    assert observation.shape == config.encoded_observation_shape
