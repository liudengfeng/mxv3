import gymnasium as gym
import gymxq
import pytest
import xqcpp
from gymxq.constants import NUM_ACTIONS

from muzero.config import MuZeroConfig
from muzero.feature_utils import obs2feature
from muzero.mcts import GameHistory
from muzero.node import Node
from muzero.replay_buffer_utils import make_target


def fake_root():
    root = Node(0)
    a = xqcpp.m2a("2838")
    root.expand([a], 1, 0, {a: 1.0}, None, True)
    child = root.children[a]
    child.reward = 1.0
    child.value = 0.0
    child.visit_count = 1

    root.visit_count = 1
    root.value_sum = 1.0
    return root


def test_rollout_and_make_target():
    init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 192"
    config = MuZeroConfig()
    game_history = GameHistory()
    environment = gym.make("xqv1", init_fen=init_fen, render_mode="ansi")
    obs, info = environment.reset()
    observation = obs2feature(obs, info, flatten=False)

    game_history.observation_history.append(observation)
    to_play = info["to_play"]
    game_history.to_play_history.append(to_play)

    action = xqcpp.m2a("2838")

    obs, reward, termination, truncation, info = environment.step(action)

    observation = obs2feature(obs, info, flatten=False)

    root = fake_root()
    game_history.store_search_statistics(root, config.action_space)

    # Next batch
    game_history.action_history.append(action)
    game_history.observation_history.append(observation)
    game_history.reward_history.append(reward)
    to_play = info["to_play"]
    game_history.to_play_history.append(to_play)
    game_history.terminated_history.append(termination)
    game_history.truncated_history.append(truncation)

    idx = 0
    target_values_0, target_rewards_0, target_policies_0, actions_0 = make_target(
        game_history, idx, config, False
    )

    assert sum(target_values_0) == 1.0
    assert target_values_0[idx] == 1.0

    assert sum(target_rewards_0) == 1.0
    assert target_rewards_0[idx + 1] == 1.0

    assert target_policies_0[idx][action] == 1.0
    for i in range(idx + 1, config.num_unroll_steps):
        assert sum(target_policies_0[i]) == 0.0

    assert actions_0[0] == NUM_ACTIONS
    assert actions_0[idx + 1] == action

    idx = 1
    target_values_1, target_rewards_1, target_policies_1, actions_1 = make_target(
        game_history, idx, config, False
    )

    assert sum(target_values_1) == 0.0

    assert sum(target_rewards_1) == 1.0
    assert target_rewards_1[0] == 1.0

    for p in target_policies_1:
        assert sum(p) == 0.0

    assert actions_1[0] == action
    for i in range(idx, config.num_unroll_steps):
        assert actions_1[i] == NUM_ACTIONS
