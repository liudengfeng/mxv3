import pytest
from muzero.node import Node, render_root
from muzero.mcts_utils import backpropagate, MinMaxStats, select_child
from muzero.config import MuZeroConfig
import numpy as np


def fake_inference_1(parent, a):
    # value, reward, policy, state
    match parent.path():
        case "root":
            match a:
                case 2086:
                    return 1, 0, {0: 0.45, 1: 0.05, 2: 0.45}, parent.path()
                case 0:
                    return 0, 1, {}, parent.children[a].path()
                case 2:
                    return 0, 1, {}, parent.children[a].path()
                case 1:
                    return 0, 0, {21: 0.5, 22: 0.5}, parent.children[a].path()
        case "root -> 0002":
            match a:
                case 21:
                    return 0, 1, {}, parent.children[a].path()
                case 22:
                    return 0, 1, {}, parent.children[a].path()


def fake_inference_2(parent, a):
    # value, reward, policy, state
    match parent.path():
        case "root":
            match a:
                case 2086:
                    return -1, 0, {0: 1.0}, parent.path()
                # 红方只有一步可走
                case 0:
                    return 0, 0, {21: 0.5, 22: 0.5}, parent.children[a].path()
        case "root -> 0001":
            match a:
                case 21:
                    return 0, 1, {}, parent.children[a].path()
                case 22:
                    return 0, 1, {}, parent.children[a].path()


def simulate(depth, actions, num, fake_func, expected_min):
    config = MuZeroConfig()
    min_max_stats = MinMaxStats()
    to_play = 1
    root = Node(0)

    value, reward, policy_logits, hidden_state = fake_func(root, 2086)
    root.expand(
        actions,
        to_play,
        reward,
        policy_logits,
        hidden_state,
        use_policy=True,
        debug=True,
    )
    root.add_exploration_noise(
        dirichlet_alpha=config.root_dirichlet_alpha,
        exploration_fraction=config.root_exploration_fraction,
    )
    for n in range(num):
        node = root
        virtual_to_play = to_play
        search_path = [node]
        while node.expanded() and node.depth() <= (depth - 1):
            action, node = select_child(node, min_max_stats, config)
            search_path.append(node)
            virtual_to_play = 1 if virtual_to_play == 2 else 2

        parent = search_path[-2]

        value, reward, policy_logits, hidden_state = fake_func(parent, action)

        node.expand(
            actions,
            virtual_to_play,
            reward,
            policy_logits,
            hidden_state,
            use_policy=True,
            debug=True,
        )

        backpropagate(
            search_path,
            value,
            virtual_to_play,
            min_max_stats,
            config,
        )
    if np.sign(expected_min) == 1:
        assert root.value() >= expected_min
    else:
        assert root.value() <= expected_min
    render_root(root, fake_func.__name__, "svg", "mcts_test_tree")
    print(min_max_stats.minimum)
    print(min_max_stats.maximum)


def test_backpropagate_1():
    # 深度为 2
    depth = 2
    actions = [0, 1, 2]
    num = 400
    fake_func = fake_inference_1
    expected_min = 0.9
    simulate(depth, actions, num, fake_func, expected_min)


def test_backpropagate_2():
    # 深度为 2
    depth = 2
    actions = [0]
    num = 240
    fake_func = fake_inference_2
    expected_min = -0.9
    simulate(depth, actions, num, fake_func, expected_min)
