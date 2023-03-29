import os
import time

import numpy as np
import ray
import torch

from .models import MuZeroNetwork
from .replay_buffer_utils import Buffer


@ray.remote
class ReplayBuffer(Buffer):
    def __init__(self, config):
        super().__init__(config)


@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, config):
        self.config = config

        # Fix random generator seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = MuZeroNetwork(self.config)
        # self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        # self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]
        self.num_reanalysed_games = 0

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while (
            ray.get(shared_storage.get_info.remote("training_step"))
            < self.config.training_steps
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            game_id, game_history, _ = ray.get(
                replay_buffer.sample_game.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            observations = np.concatenate(
                [
                    game_history.get_stacked_observations(
                        i,
                    )
                    for i in range(len(game_history.root_values))
                ]
            )

            observations = (
                torch.tensor(observations)
                .float()
                .to(next(self.model.parameters()).device)
            )

            values, _, _, _ = self.model.initial_inference(observations)

            game_history.reanalysed_predicted_root_values = (
                values.detach().cpu().numpy()
            )

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
