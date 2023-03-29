from gymxq.constants import (
    NUM_COL,
    NUM_ROW,
    NUM_PIECE,
    MAX_NUM_NO_EAT,
    NUM_PLAYER,
    NUM_ACTIONS,
)
import torch

# 棋盘特征
PLANE_NUM = 17
STACKED_NUM = 1


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (
            PLANE_NUM,
            NUM_ROW,
            NUM_COL,
        )  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(
            range(NUM_ACTIONS)
        )  # Fixed list of all possible actions. You should only edit the length
        self.players = list(
            range(1, NUM_PLAYER + 1)
        )  # List of players. You should only edit the length
        self.stacked_observations = STACKED_NUM  # Number of previous observations and previous actions to add to the current observation
        self.encoded_observation_shape = (
            STACKED_NUM,
            PLANE_NUM,
            NUM_ROW,
            NUM_COL,
        )
        self.encoded_action_shape = (2, NUM_ROW, NUM_COL)
        # Evaluate
        self.muzero_player = 1  # Turn Muzero begins to play (1: MuZero plays first, 2: MuZero plays second)
        self.opponent = 2  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 16  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        # 比较 gpu 与 cpu 速度
        self.selfplay_on_gpu = False
        self.max_moves = 200  # Maximum number of moves if game is not finished before
        # CPU串行模拟240次45s
        # GPU速度慢
        # 并行不可行
        # 初期模拟次数少
        # 使用增加模式
        # self.num_simulations = 400  # Number of future moves self-simulated 论文：800 缩小至400
        self.num_simulations = 400  # Number of future moves self-simulated
        self.discount_factor = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        # Residual Network
        # 8层 模型检查点 366M 2层 模型检查点 204M
        # self.blocks = 16  # Number of blocks in the ResNet
        self.blocks = 2  # Number of blocks in the ResNet
        # self.support_size = 1
        self.channels = 256  # Number of channels in the ResNet
        self.reduced_channels_reward = 256  # Number of channels in reward head
        self.reduced_channels_value = 256  # Number of channels in value head
        self.reduced_channels_policy = 256  # Number of channels in policy head
        self.resnet_fc_reward_layers = [
            256,
            265,
        ]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [
            265,
            265,
        ]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [
            265,
            265,
        ]  # Define the hidden layers in the policy head of the prediction network

        ### Training
        self.save_model = (
            True  # Save the checkpoint in results_path as model.checkpoint
        )
        self.training_steps = int(1e6)
        # self.steps_before_train = 100
        # Total number of training steps (ie weights update according to a batch)
        self.batch_size = (
            128  # Number of parts of games to train on at each training step
        )
        # 等待自玩对局累计步数达此标准后才进行训练
        self.steps_before_train = self.batch_size
        # 缓存N自玩对局后报告动态
        self.buffer_report_interval = 10
        # 每N次训练后保存检查点
        self.checkpoint_interval = max(
            self.training_steps // 1000, 10
        )  # save checkpoint interval
        # self.test_interval = max(self.checkpoint_interval // 2, 5)
        self.test_interval = 1000
        # 更新代理模型参数
        # self.update_model_interval = max(self.test_interval // 2, 2)
        self.update_model_interval = 1
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # 线性递减上下界学习速率
        self.lr_init = 0.003  # Initial learning rate
        self.lr_end = 0.0001  # Set it to 1 to use a constant learning rate

        ### Replay Buffer
        self.replay_buffer_size = int(
            1e6
        )  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = (
            5  # Number of game moves to keep for every batch element
        )
        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value, None for Monte Carlo
        self.PER = False  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        # self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1.5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

        # self.init_fen = "" # 初始fen字符串，正式训练时应设置为空白字符串
        self.init_fen = (
            "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 180"  # 初始fen字符串，正式训练时应设置为空白字符串
        )

        self.runs = 0  # 实验次数，用于管理存储实验数据路径
        self.debug_mcts = False  # 是否显示搜索树信息
        self.debug_duration = False  # 是否显示搜索树信息
        self.mcts_fmt = "svg"  # 搜索树文件格式 [svg,png]
        self.restore_from_latest_checkpoint = False  # 如存在最近检查点，则加载模型、优化器、缓存数据
        self.reset = False  # 除模型参数外，其余恢复至初始训练状态

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def to_dict(self) -> dict:
        d = vars(self).copy()
        size = len(d["action_space"])
        del d["action_space"]
        d["action_space_size"] = size
        return d

    def linear_lr(self, training_step):
        """线性递减上下界学习速率

        Args:
            training_step (int): 训练步数

        Returns:
            float: 学习速率
        """
        lower = int(0.1 * self.training_steps)
        upper = int(0.9 * self.training_steps)
        if training_step <= lower:
            return self.lr_init
        elif training_step >= upper:
            return self.lr_end
        return self.lr_init - (training_step - lower) / (upper - lower) * (
            self.lr_init - self.lr_end
        )
