"""Builders for distributed training.
1. 便于直观，特征暂时不编码
2. 序列环境使用简单环境包装，添加视频记录功能
"""
import gymnasium as gym
import gymxq
import numpy as np


def stack_observation(observations, infos):
    stacked = observations.copy()
    keys = ["to_play", "legal_actions", "last_action"]
    for key in keys:
        stacked[key] = infos[key]
    return stacked


class EnvironmentzBase:
    """Environments base."""

    def __init__(self, workers: int, init_fen: str):
        self.env_name = "xqv1"
        self.environments = None
        self.init_fen = init_fen
        self.workers = workers
        self.started = False

    def start(self):
        """Used once to get the initial observations."""
        assert not self.started
        self.started = True
        obs, infos = self.environments.reset()
        return stack_observation(obs, infos)

    def step(self, actions):
        (
            observations,
            rewards,
            terminations,
            truncations,
            step_infos,
        ) = self.environments.step(actions)
        resets = []
        continuous_uneaten = []
        s = []
        steps = []
        # obs_keys = ["continuous_uneaten", "s", "steps"]
        # info_keys = ["to_play", "legal_actions"]
        for i in range(self.workers):
            reset = terminations[i] or truncations[i]
            resets.append(reset)
            if reset:
                next_obs = step_infos["final_observation"][i]
                if next_obs is not None:
                    s.append(next_obs["s"])
                    steps.append(next_obs["steps"])
                    continuous_uneaten.append(next_obs["continuous_uneaten"])
            else:
                s.append(observations["s"][i])
                continuous_uneaten.append(observations["continuous_uneaten"][i])
                steps.append(observations["steps"][i])

        infos = dict(
            # next batch
            observations=dict(
                s=np.asarray(s),
                steps=np.asarray(steps),
                continuous_uneaten=np.asarray(continuous_uneaten),
                to_play=step_infos["to_play"],
                legal_actions=step_infos["legal_actions"],
            ),
            rewards=rewards,
            resets=np.array(resets, np.bool_),
            terminations=terminations,
            truncations=truncations,
        )
        return stack_observation(observations, step_infos), infos


# class Sequential(EnvironmentzBase):
#     """A group of environments used in sequence."""

#     def __init__(self, workers, init_fen):
#         assert workers == 1, "Sequential限定一个环境"
#         super().__init__(workers, init_fen)
#         self.environments = gym.vector.make(
#             self.env_name,
#             init_fen=init_fen,
#             gen_qp=True,
#             render_mode="rgb_array",
#             num_envs=workers,
#             asynchronous=False,
#         )


class Parallel(EnvironmentzBase):
    """A group of sequential environments used in parallel."""

    def __init__(self, workers: int = 1, init_fen: str = ""):
        super().__init__(workers, init_fen)
        self.environments = gym.vector.make(
            self.env_name, init_fen=init_fen, gen_qp=False, num_envs=workers
        )
