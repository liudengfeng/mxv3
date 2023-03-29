import copy
import time

import numpy as np
import ray
import torch

from .self_play import SelfPlay, SelfTestPlay
from .utils import duration_repr


@ray.remote
class SelfPlayActor(SelfPlay):
    def __init__(self, config, worker_id: int = 1):
        super().__init__(config, worker_id)

        self.num_played_games = 0
        self.num_played_steps = 0

        # Fix random generator seed
        np.random.seed(config.seed + (worker_id + 1) * 1000)
        torch.manual_seed(config.seed + (worker_id + 1) * 1000)

        self.start = time.time()

    def __repr__(self):
        return "自玩 on {}(index={:03d})".format(self.device, self.worker_id + 1)

    def continuous_self_play(self, replay_buffer, shared_storage):
        while not ray.get(shared_storage.get_info.remote("terminate")):
            training_step = ray.get(shared_storage.get_info.remote("training_step"))
            # 同步参数
            current_version = ray.get(shared_storage.get_info.remote("model_version"))
            if current_version > self.model_version:
                weights = ray.get(shared_storage.get_info.remote("weights")).copy()
                # self.model.set_weights(copy.deepcopy(weights))
                self.model.set_weights(weights)
                self.model_version = current_version
                # print(f"🚨load_weights version {current_version:03d}")

            game_history = self.rollout(training_step)
            replay_buffer.save_game.remote(game_history, shared_storage)
            self.num_played_games += 1
            self.num_played_steps += len(game_history.root_values)
            duration = time.time() - self.start
            per_game = duration / self.num_played_games
            per_step = duration / self.num_played_steps
            msg = "⏱️ Played {:>7d} games {:>9d} steps avg length {:.1f} duration {}[{:>6.2f} s/game {:>6.2f} s/step]".format(
                self.num_played_games,
                self.num_played_steps,
                self.num_played_steps / self.num_played_games,
                duration_repr(duration),
                per_game,
                per_step,
            )
            print(msg)


@ray.remote
class SelfTestPlayActor(SelfTestPlay):
    def __repr__(self):
        return "测试 on {}(index={:03d})".format(self.device, self.worker_id + 1)

    def continuous_self_play(self, shared_storage):
        while True:
            keys = ["terminate", "training_step", "model_version"]
            info = ray.get(shared_storage.get_info.remote(keys))
            if info["terminate"]:
                break
            if (
                info["training_step"] <= self.config.training_steps
                and info["training_step"] >= 1
                and info["training_step"] % self.config.test_interval == 0
            ):
                # 同步参数
                current_version = info["model_version"]
                if current_version > self.model_version:
                    # 只读亦可
                    weights = ray.get(shared_storage.get_info.remote("weights"))
                    self.model.set_weights(weights)
                    self.model_version = current_version
                    # print(f"🚨load_weights version {current_version:03d}")

                training_step = info["training_step"]
                game_history = self.rollout(training_step)
                # Save to the shared storage
                reward_dict = game_history.get_reward_pair(self.config.muzero_player)
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.root_values),
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": np.mean(
                            [value for value in game_history.root_values if value]
                        ),
                        "muzero_reward": reward_dict["muzero_reward"],
                        "opponent_reward": reward_dict["opponent_reward"],
                    }
                )
                print(
                    "🚨 episode_length {} muzero_reward {} opponent_reward {}".format(
                        len(game_history.root_values),
                        reward_dict["muzero_reward"],
                        reward_dict["opponent_reward"],
                    )
                )
