import copy
import os
import shutil
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.record_video import RecordVideo
from gymxq.constants import *

from .feature_utils import encoded_action, obs2feature
from .mcts import MCTS, GameHistory, render_root
from .models import MuZeroNetwork
from .path_utils import get_experiment_path
from .utils import duration_repr


def select_action(node, temperature):
    """
    Select action according to the visit count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function
    in the config.
    """
    visit_counts = np.array(
        [child.visit_count for child in node.children.values()], dtype="int32"
    )
    actions = [action for action in node.children.keys()]
    if temperature == 0:
        action = actions[np.argmax(visit_counts)]
    elif temperature == float("inf"):
        action = np.random.choice(actions)
    else:
        # See paper appendix Data Generation
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(
            visit_count_distribution
        )
        action = np.random.choice(actions, p=visit_count_distribution)

    return action


class BaseSelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, config, worker_id: int = 0, test_mode=False):
        self.config = config
        self.worker_id = worker_id
        self.test_mode = test_mode
        self.model_version = 0

        self.model = torch.compile(MuZeroNetwork(self.config))
        self.device = "cuda" if config.selfplay_on_gpu else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # 管理路径
        self.root_path = get_experiment_path(config.runs)

        # 管理环境
        self.environment = None

    def rollout(self, training_step):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        temperature = (
            self.config.visit_softmax_temperature_fn(training_step)
            if not self.test_mode
            else 0
        )

        game_history = GameHistory()
        obs, info = self.environment.reset()
        observation = obs2feature(obs, info, flatten=False)
        # last_a = game_history.action_history[-1]
        # observation = np.concatenate(
        #     [encoded_action(last_a)[np.newaxis, :], observation], axis=1
        # )

        game_history.observation_history.append(observation)
        to_play = info["to_play"]
        game_history.to_play_history.append(to_play)

        reset = False
        episode_steps = 0
        with torch.no_grad():
            while not reset:
                legal_actions = info["legal_actions"]
                start = time.time()
                # Choose the action
                root, mcts_info = MCTS(self.config).run(
                    self.model,
                    observation,
                    legal_actions,
                    to_play,
                    True if not self.test_mode else False,
                )

                # 演示MCTS搜索树
                if self.test_mode and self.config.debug_mcts:
                    filename = "episode_{:06d}_{:03d}".format(
                        training_step, episode_steps
                    )
                    render_root(root, filename, self.config.mcts_fmt, self.mcts_path)

                if self.config.debug_duration:
                    print(
                        "第{:>3d} 步 模拟次数 {:>3d} 最大深度 {:>3d} 根值 {:>-5.2f} 用时 {}".format(
                            episode_steps + 1,
                            self.config.num_simulations,
                            mcts_info["max_tree_depth"],
                            mcts_info["root_predicted_value"],
                            duration_repr(time.time() - start),
                        )
                    )

                action = select_action(root, temperature)

                if self.test_mode:
                    # 在step前添加AI提示信息
                    sum_visits = sum(
                        child.visit_count for child in root.children.values()
                    )
                    tips = [
                        (
                            a,
                            root.children[a].visit_count / sum_visits,
                            # 应取反 【对手状态值】
                            -root.children[a].value(),
                            root.children[a].prior,
                        )
                        for a in legal_actions
                    ]
                    self.environment.add_ai_tip(tips)

                obs, reward, termination, truncation, info = self.environment.step(
                    action
                )

                if not self.test_mode and reward == 1:
                    print(self.environment.render())
                    import xqcpp

                    hist = []
                    for a in game_history.action_history[1:] + [action]:
                        hist.append(xqcpp.a2m(a))
                    print(len(hist), hist)

                episode_steps += 1

                observation = obs2feature(obs, info, flatten=False)

                # last_a = game_history.action_history[-1]
                # observation = np.concatenate(
                #     [encoded_action(last_a)[np.newaxis, :], observation], axis=1
                # )

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                to_play = info["to_play"]
                game_history.to_play_history.append(to_play)
                game_history.terminated_history.append(termination)
                game_history.truncated_history.append(truncation)

                reset = termination or truncation

        return game_history


class SelfPlay(BaseSelfPlay):
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, config, worker_id: int = 0):
        super().__init__(config, worker_id, False)
        self.environment = gym.make(
            "xqv1", init_fen=config.init_fen, render_mode="ansi"
        )


class SelfTestPlay(BaseSelfPlay):
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, config, worker_id: int = 0):
        super().__init__(config, worker_id, True)

        # 测试环境 rgb_array 录制视频
        environment = gym.make(
            "xqv1",
            init_fen=config.init_fen,
            gen_qp=True,
            render_mode="rgb_array",
        )

        self.video_path = os.path.join(
            self.root_path,
            "{}_videos_{:03d}".format("test", worker_id),
        )
        self.mcts_path = os.path.join(
            self.root_path,
            "{}_worker_{:03d}".format("test", worker_id),
        )
        if os.path.exists(self.mcts_path):
            shutil.rmtree(self.mcts_path, ignore_errors=True)

        if os.path.exists(self.video_path):
            shutil.rmtree(self.video_path, ignore_errors=True)

        self.environment = RecordVideo(
            environment,
            self.video_path,
            episode_trigger=lambda x: True,
            disable_logger=True,
            # name_prefix="worker_{:03d}".format(worker_id),
        )

        print("🚨 测试游戏视频存放目录:{}".format(self.video_path))
        if self.config.debug_mcts:
            print("🚨 测试游戏MCTS搜索树存放目录{}".format(self.mcts_path))
