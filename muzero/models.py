import torch
from abc import ABC, abstractmethod


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class MuZeroNetwork:
    def __new__(cls, config):
        return MuZeroResidualNetwork(
            config.observation_shape,
            config.stacked_observations,
            len(config.action_space),
            config.blocks,
            config.channels,
            config.reduced_channels_reward,
            config.reduced_channels_value,
            config.reduced_channels_policy,
            config.resnet_fc_reward_layers,
            config.resnet_fc_value_layers,
            config.resnet_fc_policy_layers,
        )


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
    ):
        super().__init__()
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations - 1) + observation_shape[0],
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        k,
        reduced_channels_reward,
        fc_reward_layers,
        block_output_size_reward,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - k)
        self.bn = torch.nn.BatchNorm2d(num_channels - k)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - k) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(
            num_channels - k,
            reduced_channels_reward,
            1,
        )
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward,
            fc_reward_layers,
            1,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        x = self.conv1x1_reward(x)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        # [-1, 1]
        reward = torch.tanh(reward).squeeze()
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, 1)
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        # 🚨 [-1, 1]
        value = torch.tanh(self.fc_value(value)).squeeze()
        policy = self.fc_policy(policy)
        return policy, value


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        # support_size,
        # downsample,
    ):
        super().__init__()
        self.action_space_size = action_space_size

        block_output_size_reward = (
            reduced_channels_reward * observation_shape[1] * observation_shape[2]
        )

        block_output_size_value = (
            reduced_channels_value * observation_shape[1] * observation_shape[2]
        )

        block_output_size_policy = (
            reduced_channels_policy * observation_shape[1] * observation_shape[2]
        )

        # self.representation_network = torch.nn.DataParallel(
        #     RepresentationNetwork(
        #         observation_shape,
        #         stacked_observations,
        #         num_blocks,
        #         num_channels,
        #     )
        # )

        self.representation_network = RepresentationNetwork(
            observation_shape,
            stacked_observations,
            num_blocks,
            num_channels,
        )

        # self.dynamics_network = torch.nn.DataParallel(
        #     DynamicsNetwork(
        #         num_blocks,
        #         num_channels + 1,
        #         reduced_channels_reward,
        #         fc_reward_layers,
        #         block_output_size_reward,
        #     )
        # )
        k = 2
        self.dynamics_network = DynamicsNetwork(
            num_blocks,
            num_channels + k,
            k,
            reduced_channels_reward,
            fc_reward_layers,
            block_output_size_reward,
        )

        # self.prediction_network = torch.nn.DataParallel(
        #     PredictionNetwork(
        #         action_space_size,
        #         num_blocks,
        #         num_channels,
        #         reduced_channels_value,
        #         reduced_channels_policy,
        #         fc_value_layers,
        #         fc_policy_layers,
        #         block_output_size_value,
        #         block_output_size_policy,
        #     )
        # )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            block_output_size_value,
            block_output_size_policy,
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        expected_shape = (
            encoded_state.shape[0],
            2,
            encoded_state.shape[2],
            encoded_state.shape[3],
        )

        msg = "action期望shape={},实际为={}".format(expected_shape, action.shape)
        assert action.shape == expected_shape, msg

        x = torch.cat((encoded_state, action), dim=1)

        next_encoded_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = 0
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


########### End ResNet ###########
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)
