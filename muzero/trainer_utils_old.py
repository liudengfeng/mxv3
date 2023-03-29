import numpy as np
import torch

from .utils import get_lr_feature, get_lr_policy


def _update_weights(
    observation_batch,
    action_batch,
    target_value,
    target_reward,
    target_policy,
    weight_batch,
    gradient_scale_batch,
    model,
    optimizer,
    config,
):
    """
    Perform one training step.
    """
    device = next(model.parameters()).device

    target_value_scalar = np.array(target_value.copy(), dtype="float32")

    priorities = np.zeros_like(target_value_scalar)

    if config.PER:
        weight_batch = torch.tensor(weight_batch.copy()).float().to(device)

    observation_batch = torch.tensor(observation_batch.copy()).float().to(device)

    action_batch = torch.tensor(action_batch.copy()).float().to(device)
    target_value = torch.tensor(target_value.copy()).float().to(device)
    target_reward = torch.tensor(target_reward.copy()).float().to(device)
    target_policy = torch.tensor(target_policy.copy()).float().to(device)
    gradient_scale_batch = torch.tensor(gradient_scale_batch.copy()).float().to(device)
    # observation_batch: batch, channels, height, width
    # action_batch: batch, num_unroll_steps+1, 2, height, width
    # target_value: batch, num_unroll_steps+1, 1
    # target_reward: batch, num_unroll_steps+1
    # target_policy: batch, num_unroll_steps+1, len(action_space)
    # gradient_scale_batch: batch, num_unroll_steps+1

    ## Generate predictions
    value, reward, policy_logits, hidden_state = model.initial_inference(
        observation_batch
    )
    predictions = [(value, reward, policy_logits)]

    for i in range(1, action_batch.shape[1]):
        value, reward, policy_logits, hidden_state = model.recurrent_inference(
            hidden_state, action_batch[:, i]
        )
        # Scale the gradient at the start of the dynamics function (See paper appendix Training)
        hidden_state.register_hook(lambda grad: grad * 0.5)
        predictions.append((value, reward, policy_logits))

    ## Compute losses
    value_loss, reward_loss, policy_loss = (0, 0, 0)
    value, reward, policy_logits = predictions[0]
    # Ignore reward loss for the first batch step
    (
        current_value_loss,
        current_reward_loss,
        current_policy_loss,
    ) = loss_function(
        value,
        reward,
        policy_logits,
        target_value[:, 0],
        target_reward[:, 0],
        target_policy[:, 0],
    )
    value_loss += current_value_loss
    policy_loss += current_policy_loss

    priorities[:, 0] = (
        np.abs(value.detach().cpu().numpy() - target_value_scalar[:, 0])
        ** config.PER_alpha
    )

    for i in range(1, len(predictions)):
        value, reward, policy_logits = predictions[i]
        (
            current_value_loss,
            current_reward_loss,
            current_policy_loss,
        ) = loss_function(
            value,
            reward,
            policy_logits,
            target_value[:, i],
            target_reward[:, i],
            target_policy[:, i],
        )
        # Scale gradient by the number of unroll steps (See paper appendix Training)
        current_value_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
        current_reward_loss.register_hook(
            lambda grad: grad / gradient_scale_batch[:, i]
        )
        # Reward Loss is omitted for board games
        current_policy_loss.register_hook(
            lambda grad: grad / gradient_scale_batch[:, i]
        )
        value_loss += current_value_loss
        reward_loss += current_reward_loss
        policy_loss += current_policy_loss

        priorities[:, i] = (
            np.abs(value.detach().cpu().numpy() - target_value_scalar[:, i])
            ** config.PER_alpha
        )

    # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
    loss = value_loss * config.value_loss_weight + reward_loss + policy_loss

    if config.PER:
        # Correct PER bias by using importance-sampling (IS) weights
        loss *= weight_batch
    # Mean over batch dimension (pseudocode do a sum)
    loss = loss.mean()

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (
        priorities,
        # For log purpose
        loss.detach().cpu().numpy().item(),
        value_loss.mean().detach().cpu().numpy().item(),
        reward_loss.mean().detach().cpu().numpy().item(),
        policy_loss.mean().detach().cpu().numpy().item(),
    )


def update_weights_lr(batch, model, optimizer, config):
    """
    Perform one training step.
    """
    (
        observation_batch,
        action_batch,
        target_value,
        target_reward,
        target_policy,
        weight_batch,
        gradient_scale_batch,
    ) = batch

    # 处理观察特征
    observation_batch = np.concatenate(observation_batch)
    observation_batch = get_lr_feature(observation_batch)

    # 编码的移动只需要左右互换即可
    action_batch = np.array(action_batch)

    action_batch = get_lr_feature(action_batch)

    target_value = np.array(target_value)
    target_reward = np.array(target_reward)

    # 政策左右互换
    b = len(target_policy)
    lr_target_policy = []
    for i in range(b):
        b_policy = []
        for p in target_policy[i]:
            b_policy.append(get_lr_policy(p))
        lr_target_policy.append(b_policy)
    target_policy = np.array(lr_target_policy)

    gradient_scale_batch = np.array(gradient_scale_batch)

    _update_weights(
        observation_batch,
        action_batch,
        target_value,
        target_reward,
        target_policy,
        weight_batch,
        gradient_scale_batch,
        model,
        optimizer,
        config,
    )


def update_weights(batch, model, optimizer, config):
    """
    Perform one training step.
    """
    (
        observation_batch,
        action_batch,
        target_value,
        target_reward,
        target_policy,
        weight_batch,
        gradient_scale_batch,
    ) = batch

    observation_batch = np.concatenate(observation_batch)

    action_batch = np.array(action_batch)

    target_value = np.array(target_value)
    target_reward = np.array(target_reward)
    target_policy = np.array(target_policy)

    # priorities = np.zeros_like(target_value_scalar)

    gradient_scale_batch = np.array(gradient_scale_batch)

    return _update_weights(
        observation_batch,
        action_batch,
        target_value,
        target_reward,
        target_policy,
        weight_batch,
        gradient_scale_batch,
        model,
        optimizer,
        config,
    )


def loss_function(
    value,
    reward,
    policy_logits,
    target_value,
    target_reward,
    target_policy,
):
    if isinstance(reward, torch.Tensor):
        reward_loss_fn = torch.nn.MSELoss(reduction="none")
        # (batch,)
        reward_loss = reward_loss_fn(reward, target_reward)
    else:
        reward_loss = 0
    value_loss_fn = torch.nn.MSELoss(reduction="none")
    # (batch,)
    value_loss = value_loss_fn(value, target_value)
    # policy_logits Predicted unnormalized logits
    # (batch,)
    policy_loss = torch.nn.functional.cross_entropy(
        policy_logits, target_policy, reduction="none"
    )
    return value_loss, reward_loss, policy_loss


def update_lr(training_step, optimizer, config):
    """
    Update learning rate
    """
    # lr = config.lr_init * config.lr_decay_rate ** (
    #     training_step / config.lr_decay_steps
    # )
    lr = config.linear_lr(training_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
