import math

import numpy as np


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def ucb_score_fn(parent, child, min_max_stats, config):
    """
    The score for a node is based on its value, plus an exploration bonus based on the prior.
    """
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior

    if child.visit_count > 0:
        # Mean value Q
        value_score = min_max_stats.normalize(
            child.reward
            + config.discount_factor
            * (child.value() if len(config.players) == 1 else -child.value())
        )
    else:
        value_score = 0

    return prior_score + value_score


def select_child(node, min_max_stats, config):
    # 简化计算量
    kvs = {
        action: ucb_score_fn(node, child, min_max_stats, config)
        for action, child in node.children.items()
    }
    # 更新ucb得分
    for action, _ in kvs.items():
        node.children[action].ucb_score = kvs[action]

    max_ucb = sorted(kvs.values())[-1]
    # 可能有多个相同的最大值
    action = np.random.choice(
        [action for action, value in kvs.items() if value == max_ucb]
    )
    return action, node.children[action]


def backpropagate(search_path: list, value: float, to_play: int, min_max_stats, config):
    """
    At the end of a simulation, we propagate the evaluation all the way up the tree
    to the root.
    """
    # 调整为玩家角度奖励
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.reward + config.discount_factor * -node.value())
        value = (
            -node.reward if node.to_play == to_play else node.reward
        ) + config.discount_factor * value
