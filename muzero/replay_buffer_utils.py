import os
import pickle
import time

import numpy as np
from gymxq.constants import NUM_ACTIONS

from .feature_utils import encoded_action
from .path_utils import get_experiment_path
from .utils import duration_repr


def sample_position(game_history, config, force_uniform=False):
    """
    Sample position from game either uniformly or according to some priority.
    See paper appendix Training.
    """
    position_prob = None
    if config.PER and not force_uniform:
        position_probs = game_history.priorities / sum(game_history.priorities)
        position_index = np.random.choice(len(position_probs), p=position_probs)
        position_prob = position_probs[position_index]
    else:
        position_index = np.random.choice(len(game_history.root_values))
    return position_index, position_prob


def compute_target_value(game_history, index: int, config):
    """计算目标值
    Args:
        game_history (GameHistory): 游戏历史实例
        index (int): 计算对象所在的序号
        config (MuZeroConfig): 配置实例
    Returns:
        float: 状态函数值
    Notes:
        state   s0   -> s1 -> s2 ... \n
        action  a0   -> a1 -> a2 ... \n
        reward  0    -> r1 -> r2 ... \n
        value   v0   <- r0 <- r1 ... \n
    """
    # The value target is the discounted root value of the search tree td_steps into the
    # future, plus the discounted sum of all rewards until then.
    bootstrap_index = index + config.td_steps
    # Use Monte Carlo for Board Games
    if bootstrap_index < len(game_history.root_values):
        root_values = (
            game_history.root_values
            if game_history.reanalysed_predicted_root_values is None
            else game_history.reanalysed_predicted_root_values
        )
        last_step_value = (
            root_values[bootstrap_index]
            if game_history.to_play_history[bootstrap_index]
            == game_history.to_play_history[index]
            else -root_values[bootstrap_index]
        )
        # 如非自然终止，其值不具有意义
        # 极端例子：下一步棋局为定式可获胜，但此时已经超限，不可使用`next state`根值
        truncated = float(game_history.truncated_history[bootstrap_index])
        value = (
            last_step_value
            * (1 - truncated)
            * config.discount_factor**config.td_steps
        )
    else:
        value = 0

    # 🚨 注意 index + 1 开始
    for i, reward in enumerate(
        game_history.reward_history[index + 1 : bootstrap_index + 1]
    ):
        # The value is oriented from the perspective of the current player
        current_reward = (
            reward
            if game_history.to_play_history[index]
            == game_history.to_play_history[index + i]
            else -reward
        )
        # 🚨 注意偏移 1
        truncated = float(game_history.truncated_history[index + i + 1])
        value += (1 - truncated) * current_reward * config.discount_factor**i
    return value


def make_target(game_history, state_index, config, encode_actions=True):
    """
    Generate targets for every unroll steps.
    """
    target_values, target_rewards, target_policies, actions = [], [], [], []
    num = len(game_history.root_values)
    last_reward = game_history.reward_history[num]
    last_action = game_history.action_history[num]

    # last_action = 1.0
    pi = [0] * len(game_history.child_visits[0])
    pi[last_action] = 1.0

    # 正常终止状态适用
    last_value = (
        game_history.root_values[num - 1] if game_history.terminated_history[num] else 0
    )

    for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
        value = compute_target_value(game_history, current_index, config)
        if current_index < num:
            target_values.append(value)
            target_rewards.append(game_history.reward_history[current_index])
            target_policies.append(game_history.child_visits[current_index])
            actions.append(game_history.action_history[current_index])
        # 吸收状态
        else:
            # 方案
            if current_index == num:
                target_values.append(last_value)
                target_rewards.append(last_reward)
            else:
                # tgt_value = last_value / config.discount_factor ** (
                #     current_index - state_index - 1
                # )
                # target_values.append(tgt_value)
                target_values.append(0)
                target_rewards.append(0)
            target_policies.append(pi)
            actions.append(last_action)
    if encode_actions:
        # a -> (2,10,9)
        actions = [encoded_action(a) for a in actions]
    return target_values, target_rewards, target_policies, actions


def update_gamehistory_priorities(game_history, config):
    """更新`GameHistory`优先度
    Args:
        game_history (GameHistory): 游戏历史对象
        config (MuZeroConfig): 配置对象
    """
    priorities = []
    for i, root_value in enumerate(game_history.root_values):
        priority = (
            np.abs(root_value - compute_target_value(game_history, i, config))
            ** config.PER_alpha
        )
        priorities.append(priority)
    # 防止0除
    game_history.priorities = np.array(priorities, dtype="float32") + 1e-8
    game_history.game_priority = np.max(game_history.priorities)


class Buffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, config):
        self.config = config
        self.buffer = {}
        self.num_played_games = 0
        self.num_played_steps = 0
        self.total_samples = 0
        self.start = time.time()

        if config.restore_from_latest_checkpoint:
            root = get_experiment_path(config.runs)
            f_path = os.path.join(root, "buffer.pkl")
            if os.path.exists(f_path):
                self.load_buffer(f_path)
                print("Load buffer from {}".format(f_path))

    def is_ready(self):
        return self.num_played_steps >= self.config.steps_before_train

    def save_buffer(self):
        path = os.path.join(get_experiment_path(self.config.runs), "buffer.pkl")
        data = {
            "buffer": self.buffer,
            "num_played_games": self.num_played_games,
            "num_played_steps": self.num_played_steps,
            "total_samples": self.total_samples,
        }
        pickle.dump(data, open(path, "wb"))
        print(f"缓存数据存储路径:\n{path}")

    def load_buffer(self, file_path):
        # path = os.path.join(get_experiment_path(runs), "buffer.pkl")
        with open(file_path, "rb") as f:
            buffer_infos = pickle.load(f)
        self.buffer = buffer_infos["buffer"]
        self.num_played_games = buffer_infos["num_played_games"]
        self.num_played_steps = buffer_infos["num_played_steps"]
        self.total_samples = buffer_infos["total_samples"]

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = np.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                update_gamehistory_priorities(game_history, self.config)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote(
                {
                    "num_played_games": self.num_played_games,
                    "num_played_steps": self.num_played_steps,
                }
            )

        if (
            self.num_played_games > 0
            and self.num_played_games % self.config.buffer_report_interval == 0
        ):
            duration = time.time() - self.start
            per_game = duration / self.num_played_games
            per_step = duration / self.num_played_steps
            msg = "⏱️ Saved {:>7d} games {:>9d} steps  duration {}[{:>6.2f} s/game {:>6.2f} s/step]".format(
                self.num_played_games,
                self.num_played_steps,
                duration_repr(duration),
                per_game,
                per_step,
            )
            print(msg)

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])

        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(
            self.config.batch_size
        ):
            game_pos, pos_prob = sample_position(game_history, self.config)

            values, rewards, policies, actions = make_target(
                game_history, game_pos, self.config
            )

            index_batch.append([game_id, game_pos])
            # 4d
            observation_batch.append(game_history.get_stacked_observations(game_pos))

            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = np.array(weight_batch, dtype="float32") / max(weight_batch)

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 2, height, width
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = np.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= np.sum(game_probs)
            game_index = np.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = np.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            game_probs = np.array(game_probs, dtype="float32")
            game_probs /= np.sum(game_probs)
            game_prob_dict = dict(
                [(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)]
            )
            selected_games = np.random.choice(game_id_list, n_games, p=game_probs)
        else:
            selected_games = np.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]
        return ret

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = np.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def get_buffer(self):
        return self.buffer

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = np.max(
                    self.buffer[game_id].priorities
                )
