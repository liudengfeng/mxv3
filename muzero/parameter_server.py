import numpy as np
import ray
import torch

from .models import MuZeroNetwork
from .checkpoint import get_current_checkpoint


@ray.remote
class ParameterServer(object):
    def __init__(self, config):
        # 确保复现
        # Fix random generator seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.model = torch.compile(MuZeroNetwork(config))
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr_init,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        # TODO
        if config.restore_from_latest_checkpoint:
            # 恢复到最近检查点
            checkpoint = get_current_checkpoint(config)
            weights = checkpoint.get("weights", None)
            if weights is not None:
                self.model.set_weights(weights)

            optimizer_state = checkpoint["optimizer_state"]
            if optimizer_state is not None:
                print("Loading optimizer...\n")
                self.optimizer.load_state_dict(optimizer_state)

            lr = checkpoint["lr"]
            if lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()
