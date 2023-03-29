import copy
import os
import time

import numpy as np
import ray
import torch

# from .models import support_to_scalar
from .models import MuZeroNetwork, dict_to_cpu
from .trainer_utils import update_lr, update_weights, update_weights_lr


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, config):
        self.config = config

        # # Fix random generator seed
        # np.random.seed(self.config.seed)
        # torch.manual_seed(self.config.seed)

        self.model_version = 0

        # Initialize the network
        self.model = MuZeroNetwork(self.config)
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model = torch.compile(self.model)
        self.model.train()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr_init,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        self.training_step = 1

    def __repr__(self):
        return "训练"

    def continuous_update_weights(self, replay_buffer, shared_storage, auto_lr=False):
        # auto_lr 自动使用原始数据生成左右互换的数据，重复训练
        keys = ["training_step", "weights", "optimizer_state"]
        info = ray.get(shared_storage.get_info.remote(keys))

        self.training_step = info["training_step"]

        self.model.set_weights(info["weights"].copy())

        optimizer_state = info["optimizer_state"]
        if optimizer_state is not None:
            print("Loading optimizer...")
            self.optimizer.load_state_dict(copy.deepcopy(optimizer_state))

        # Wait for the replay buffer to be filled
        while not ray.get(replay_buffer.is_ready.remote()):
            # 间隔拉长
            time.sleep(1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while True:
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()

            update_lr(self.training_step, self.optimizer, self.config)

            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = update_weights(batch, self.model, self.optimizer, self.config)

            # 使用左右互换的数据来更新模型参数
            update_weights_lr(batch, self.model, self.optimizer, self.config)

            self.training_step += 1

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Update agent model
            if self.training_step % self.config.update_model_interval == 0:
                self.model_version += 1
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "model_version": self.model_version,
                    }
                )

            # Save checkpoint or model
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                # 覆盖式存储
                shared_storage.save_checkpoint.remote()
                replay_buffer.save_buffer.remote()

            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            print(
                "{:>7d}nth loss [total:{:>6.2f} value:{:>6.2f} reward:{:>6.2f} policy:{:>6.2f}] lr:{:.5f}".format(
                    self.training_step,
                    total_loss,
                    value_loss,
                    reward_loss,
                    policy_loss,
                    self.optimizer.param_groups[0]["lr"],
                )
            )

            if self.training_step > self.config.training_steps + 1:
                break

        shared_storage.set_info.remote("terminate", True)
        # 保存检查点
        shared_storage.set_info.remote(self.config.to_dict())
        shared_storage.save_checkpoint.remote()
