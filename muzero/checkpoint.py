import torch
import os
import numpy as np
import copy

from .models import MuZeroNetwork
from .path_utils import get_experiment_path

DEFAULS = {
    "optimizer_state": None,
    "model_version": 1,
    "total_reward": 0,
    "muzero_reward": 0,
    "opponent_reward": 0,
    "episode_length": 0,
    "mean_value": 0,
    "training_step": 0,
    "lr": None,
    "total_loss": 0,
    "value_loss": 0,
    "reward_loss": 0,
    "policy_loss": 0,
    "num_played_games": 0,
    "num_played_steps": 0,
    "num_reanalysed_games": 0,
    "terminate": False,
}


def get_initial_checkpoint(config):
    """获取初始检查点【含模型参数】

    Args:
        config : 配置对象

    Returns:
        dict: 检查点字典
    """
    checkpoint = {}
    model = MuZeroNetwork(config)
    weights = model.get_weights()
    checkpoint["weights"] = weights
    for k, v in DEFAULS.items():
        checkpoint[k] = v
    return checkpoint


def get_current_checkpoint(config):
    """当前最新检查点"""
    if config.restore_from_latest_checkpoint:
        current_checkpoint = {}
        root = get_experiment_path(config.runs)
        checkpoint_path = os.path.join(root, "model.checkpoint")
        if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            key = "weights"
            current_checkpoint[key] = copy.deepcopy(checkpoint[key])
            if not config.reset:
                for k in checkpoint.keys():
                    if k != key:
                        current_checkpoint[k] = checkpoint[k]
            else:
                print("除模型参数外，其余指标恢复至默认值")
                for k in checkpoint.keys():
                    if k == key:
                        continue
                    elif k in DEFAULS.keys():
                        current_checkpoint[k] = DEFAULS.get(k, None)
                current_checkpoint.update(config.to_dict())
            return current_checkpoint
        else:
            return get_initial_checkpoint(config)
    else:
        return get_initial_checkpoint(config)
