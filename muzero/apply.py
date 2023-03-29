import gymnasium as gym
import gymxq
import numpy as np
import torch
import xqcpp
from gymxq.constants import NUM_ACTIONS

from .checkpoint import get_current_checkpoint
from .feature_utils import encoded_action, obs2feature
from .mcts import MCTS, render_root
from .models import MuZeroNetwork
from .self_play import select_action


def get_pretrained_model(config):
    """训练模型

    Args:
        config (配置): 配置对象

    Returns:
        model: 模型
    """
    model = MuZeroNetwork(config)
    checkpoint = get_current_checkpoint(config)
    model.set_weights(checkpoint["weights"])
    print("model version = {:>7d}".format(checkpoint["model_version"]))
    model = torch.compile(model)
    model.eval()
    return model


def initial_inference(config):
    model = get_pretrained_model(config)
    environment = gym.make("xqv1", init_fen=config.init_fen)
    obs, infos = environment.reset()
    observation = obs2feature(obs, infos, flatten=False)
    # last_a = NUM_ACTIONS
    # observation = np.concatenate(
    #     [encoded_action(last_a)[np.newaxis, :], observation], axis=1
    # )
    observation = torch.tensor(observation).to(next(model.parameters()).device)
    with torch.no_grad():
        (
            root_predicted_value,
            reward,
            policy_logits,
            hidden_state,
        ) = model.initial_inference(observation)
    policy_values = torch.softmax(
        torch.tensor([policy_logits[0][a] for a in infos["legal_actions"]]), dim=0
    ).tolist()
    policy = {
        xqcpp.a2m(a): round(policy_values[i], 2)
        for i, a in enumerate(infos["legal_actions"])
    }
    return {
        "policy": policy,
        "value": round(root_predicted_value.detach().cpu().numpy().item(), 2),
    }


def get_muzero_action(config, moves=[], debug=True):
    config.debug_mcts = True
    model = get_pretrained_model(config)
    environment = gym.make("xqv1", init_fen=config.init_fen)
    obs, infos = environment.reset()
    for m in moves:
        _, _, _, _, infos = environment.step(xqcpp.m2a(m))
    legal_actions = infos["legal_actions"]
    legal_moves = []
    for a in legal_actions:
        legal_moves.append(xqcpp.a2m(a))
    print("合法移动", legal_moves)
    to_play = infos["to_play"]
    observation = obs2feature(obs, infos, flatten=False)
    root, mcts_info = MCTS(config).run(
        model,
        observation,
        legal_actions,
        to_play,
        False,
    )
    action = select_action(root, 0)
    if debug:
        filename = "episode_{:06d}_{:03d}".format(1, 1)
        render_root(root, filename, "svg", "demo_mcts_tree")
        sum_visits = sum(child.visit_count for child in root.children.values())
        tips = [
            (
                xqcpp.a2m(a),
                round(root.children[a].visit_count / sum_visits, 2),
                # 应取反 【对手状态值】
                round(-root.children[a].value(), 2),
                round(root.children[a].prior, 2),
            )
            for a in legal_actions
        ]
        tips = sorted(tips, key=lambda x: x[1], reverse=True)
        print(tips)
        print("root value = {:.2f}".format(round(root.value(), 2)))
    return action
