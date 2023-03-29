import copy
import os

import ray
import torch

from .checkpoint import get_current_checkpoint
from .path_utils import get_experiment_path


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, config):
        self.config = config
        self.current_checkpoint = get_current_checkpoint(config)

    def save_checkpoint(self):
        path = os.path.join(get_experiment_path(self.config.runs), "model.checkpoint")
        to_save = self.current_checkpoint.copy()
        to_save["terminate"] = False
        torch.save(to_save, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def incr(self, key):
        self.current_checkpoint[key] += 1

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    # def incr(self, keys, values=None):
    #     if isinstance(keys, str) and values is not None:
    #         self.current_checkpoint[keys] += values
    #     elif isinstance(keys, dict):
    #         for k, v in keys.items():
    #             self.current_checkpoint[k] += v

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
