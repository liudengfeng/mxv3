import os

# # This forces OpenMP to use 1 single thread, which is needed to
# # prevent contention between multiple actors.
# # See https://docs.ray.io/en/latest/ray-core/configure.html for
# # more details.
os.environ["OMP_NUM_THREADS"] = "1"
# # Tell numpy to only use one core. If we don't do this, each actor may
# # try to use all of the cores and the resulting contention may result
# # in no speedup over the serial version. Note that if numpy is using
# # OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
# # probably need to do it from the command line (so it happens before
# # numpy is imported).
os.environ["MKL_NUM_THREADS"] = "1"

# # To record callsite information for each ObjectRef created
# os.environ["RAY_record_ref_creation_sites"] = "1"


import argparse
import logging
import math
import pprint
import time

import coloredlogs
import ray
from muzero.utils import duration_repr
from muzero.config import MuZeroConfig
from muzero.replay_buffer import ReplayBuffer, Reanalyse
from muzero.shared_storage import SharedStorage
from muzero.trainer import Trainer
from muzero.actors import SelfPlayActor, SelfTestPlayActor


def train(logger, config: MuZeroConfig):
    gpu_on_trainer = 1.0
    gpu_on_selfplayer = 0.0
    if config.selfplay_on_gpu:
        gpu_on_trainer = 0.20
        total = config.num_workers
        if config.use_test:
            total += 1
        if config.use_reanalyse:
            total += 1
        # 略微存在损失
        gpu_on_selfplayer = math.floor((1 - gpu_on_trainer) / total * 1000) / 1000.0

    logger.info(f"每个自玩分配 {gpu_on_selfplayer:.3f} GPU，训练器分配 {gpu_on_trainer:.3f} GPU")

    shared_storage = SharedStorage.remote(config)
    replay_buffer = ReplayBuffer.remote(config)

    test_worker = SelfTestPlayActor.options(num_gpus=gpu_on_selfplayer).remote(config)

    if config.use_reanalyse:
        reanalyse_worker = Reanalyse.options(num_gpus=gpu_on_selfplayer).remote(config)

    self_players = [
        SelfPlayActor.options(num_gpus=gpu_on_selfplayer).remote(config, i)
        for i in range(config.num_workers)
    ]

    # GPU 分配给训练器
    trainer = Trainer.options(num_gpus=gpu_on_trainer).remote(config)

    result_refs = [
        trainer.continuous_update_weights.remote(replay_buffer, shared_storage)
    ]

    if config.use_test:
        result_refs.append(test_worker.continuous_self_play.remote(shared_storage))
        logger.info("use test")

    if config.use_reanalyse:
        result_refs.append(
            reanalyse_worker.reanalyse.remote(replay_buffer, shared_storage)
        )
        logger.info("use reanalyse")

    for worker in self_players:
        result_refs.append(
            worker.continuous_self_play.remote(replay_buffer, shared_storage)
        )

    ray.get(result_refs)


if __name__ == "__main__":
    # Lets gather arguments
    rem_steps = 10
    parser = argparse.ArgumentParser(description="MuZero Pytorch Implementation")
    parser.add_argument(
        "--init_fen",
        type=str,
        # default="3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190",
        # 车马兵类
        default=f"2b6/3ka4/2Pa5/3N5/9/3R5/9/9/5pr2/3AK4 r - {120-rem_steps} 0 {200-rem_steps}",
        # 里应外合
        # default=f"3k1a3/4PP3/3a5/9/9/9/5C3/3A1p3/3p1pr2/4K4 r - {120-rem_steps} 0 {200-rem_steps}",
        # 【2步杀】
        # default="2r2k3/6R1C/b4N1rb/9/5n3/5C3/6n2/5p3/4p4/5K1R1 r - 110 0 180",
        # 双兵胜双士
        # default="5a3/3ka4/2P6/5P3/9/9/9/9/9/4K4 r - 110 0 180",
        # 单马杀
        # default="3k5/9/9/4p4/9/8N/9/9/9/4K4 r - 100 0 100",
        help="初始fen字符串 (default: %(default)s)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        # default=os.cpu_count() - 2,
        default=10,
        help="Number of self play actors running concurrently (default: %(default)s)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Experiment serial number (default: %(default)s)",
    )
    parser.add_argument(
        "--selfplay_on_gpu",
        action="store_true",
        default=False,
        help="自玩是否使用GPU (default: %(default)s)",
    )
    # 恢复最近数据
    parser.add_argument(
        "--restore_from_latest_checkpoint",
        action="store_true",
        default=True,
        help="是否使用最近检查点模型参数、缓存数据 (default: %(default)s)",
    )
    # 临时使用的参数
    parser.add_argument(
        "--training_steps",
        type=int,
        default=100000,
        help="Experiment serial number (default: %(default)s)",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=240,
        help="Experiment serial number (default: %(default)s)",
    )
    parser.add_argument(
        "--use_test",
        action="store_true",
        default=True,
        help="训练中是否测试。(default: %(default)s)",
    )
    parser.add_argument(
        "--use_reanalyse",
        action="store_true",
        default=False,
        help="训练中是否再分析。(default: %(default)s)",
    )
    parser.add_argument(
        "--debug_mcts",
        action="store_true",
        default=False,
        help="是否显示MCTS搜索树。正式环境下应设置`False` (default: %(default)s)",
    )
    parser.add_argument(
        "--debug_duration",
        action="store_true",
        default=False,
        help="是否显示每次MCTS搜索用时。正式环境下应设置`False` (default: %(default)s)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=True,
        help="除模型参数外，其余指标恢复至默认值。正式环境下应设置`False` (default: %(default)s)",
    )
    args = parser.parse_args()
    coloredlogs.install(fmt="%(asctime)s[%(levelname)s] > %(message)s", level="DEBUG")
    config = MuZeroConfig()

    for k, v in vars(args).items():
        setattr(config, k, v)
    pp = pprint.PrettyPrinter(indent=4)
    logger = logging.getLogger("muzero")
    logger.info("基础配置\n")
    pp.pprint(config.to_dict())

    try:
        start = time.time()
        ray.init(num_gpus=1)
        # 引用对象
        train(logger, config)
        logger.info("Done! duration {}".format(duration_repr(time.time() - start)))
    except KeyboardInterrupt:
        logger.warning("提前终止\n")
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        ray.shutdown()
