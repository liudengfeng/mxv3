import gymnasium as gym
import gymxq
import pytest
import xqcpp
from muzero.feature_utils import obs2feature, ps2feature
from muzero.config import MuZeroConfig
import numpy as np


def make_env(fen):
    environment = gym.make("xqv1", init_fen=fen, render_mode="ansi")
    return environment


def test_ps2feature():
    init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 192"
    environment = make_env(init_fen)
    obs, info = environment.reset()
    f = ps2feature(obs["s"])
    np.testing.assert_equal(np.sum(f), 6.0)
    # 黑卒
    np.testing.assert_equal(np.sum(f[0]), 2.0)
    # 红兵
    np.testing.assert_equal(np.sum(f[0 + 7]), 2.0)

    # 将
    np.testing.assert_equal(np.sum(f[6]), 1.0)
    # 帅
    np.testing.assert_equal(np.sum(f[6 + 7]), 1.0)


def test_env_features():
    init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 192"
    config = MuZeroConfig()
    environment = make_env(init_fen)
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


def test_lr_features():
    init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 192"
    init_fen_lr = "5k3/4P1P2/9/9/9/9/9/9/2p1p4/3K5 r - 110 0 192"
    environment = make_env(init_fen)
    environment_lr = make_env(init_fen_lr)
    obs, info = environment.reset()
    observation_lr = obs2feature(obs, info, flatten=False, lr=True)

    # 使用左右互换后的棋盘
    obs_lr, info_lr = environment_lr.reset()
    observation_lr_actual = obs2feature(obs_lr, info_lr, flatten=False)
    np.testing.assert_array_equal(observation_lr, observation_lr_actual)

    # for i in range(19):
    #     print("\n", observation_lr[0][i])
    #     print("\n", observation_lr_actual[0][i])
    #     print(observation_lr[0][i] - observation_lr_actual[0][i])
    #     print("=" * 30)
