from typing import List

import numpy as np
import torch
from gymxq.constants import BLACK_PLAYER, NUM_ACTIONS, RED_PLAYER

from .feature_utils import encoded_action
from .mcts_utils import MinMaxStats, backpropagate, select_child
from .node import Node


def inference(model, observations, action):
    device = next(model.parameters()).device
    if not isinstance(observations, torch.Tensor):
        observations = np.array(observations)
        observations = torch.tensor(observations).float()
    observations = observations.to(device)
    actions = np.array([encoded_action(action)])
    actions = torch.tensor(actions).float().to(device)
    return model.inference(observations, actions)


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        root = Node(0)
        assert to_play in (RED_PLAYER, BLACK_PLAYER), "玩家编码必须为{}".format(
            (RED_PLAYER, BLACK_PLAYER)
        )

        assert (
            observation.shape == self.config.encoded_observation_shape
        ), "Observation shape should be {}".format(
            self.config.encoded_observation_shape
        )

        (
            root_predicted_value,
            reward,
            policy_logits,
            hidden_state,
        ) = inference(model, observation, NUM_ACTIONS)

        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."

        root.expand(
            legal_actions,
            to_play,
            reward,
            policy_logits,
            hidden_state,
            debug=True if self.config.debug_mcts and self.config.use_test else False,
        )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0

        for n in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            # 限制深度
            while node.expanded() and current_tree_depth < self.config.max_moves:
                current_tree_depth += 1
                action, node = select_child(node, min_max_stats, self.config)
                search_path.append(node)

                # Players play turn by turn
                virtual_to_play = (
                    RED_PLAYER if virtual_to_play == BLACK_PLAYER else BLACK_PLAYER
                )

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]

            value, reward, policy_logits, hidden_state = inference(
                model,
                parent.hidden_state,
                action,
            )

            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward.detach().cpu().numpy().item(),
                policy_logits,
                hidden_state,
                debug=True if self.config.debug_mcts else False,
            )

            backpropagate(
                search_path,
                value.detach().cpu().numpy().item(),
                virtual_to_play,
                min_max_stats,
                self.config,
            )

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.terminated_history = []
        self.truncated_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

        # 🚨对齐
        self.reward_history.append(0)
        self.action_history.append(NUM_ACTIONS)
        self.terminated_history.append(False)
        self.truncated_history.append(False)

    def get_stacked_observations(self, index):
        """
        Generate a new observation with the observation at the index position
        """
        n = len(self.observation_history)
        valid_idx = [-1] + list(range(n))
        assert index in valid_idx, "有效索引为{},输入{}".format(valid_idx, index)
        if index == -1:
            start = n - 1
        else:
            start = index
        return self.observation_history[start].copy()

    def store_search_statistics(self, root: Node, action_space: List[int]):
        """为游戏历史对象存储统计信息
        Args:
            root (Node): 根节点
            action_space (List[int]): 整数移动空间列表
        """
        # 将访问次数转换为政策
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def get_reward_pair(self, muzero_player):
        # 象棋只有最终结果可能非0
        idx = len(self.reward_history) - 1
        if self.to_play_history[idx - 1] == muzero_player:
            return {
                "muzero_reward": self.reward_history[idx],
                "opponent_reward": 0,
            }
        else:
            return {
                "muzero_reward": 0,
                "opponent_reward": self.reward_history[idx],
            }
