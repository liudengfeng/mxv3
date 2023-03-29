import numpy as np
import xqcpp
from gymxq.constants import (
    MAX_EPISODE_STEPS,
    MAX_NUM_NO_EAT,
    NUM_ACTIONS,
    NUM_COL,
    NUM_PIECE,
    NUM_PLAYER,
    NUM_ROW,
    RED_PLAYER,
    BLACK_PLAYER,
)
from gymxq.utils import move_to_coordinate, render_board_to_text

from .config import PLANE_NUM

# 棋子编码
PIECE_MAPS = {
    7: "K",
    -7: "k",
    6: "R",
    -6: "r",
    5: "N",
    -5: "n",
    4: "C",
    -4: "c",
    3: "B",
    -3: "b",
    2: "A",
    -2: "a",
    1: "P",
    -1: "p",
    0: "+",
}


def get_encoded_piece(x):
    """非空棋子编码
    Args:
        x (int): 棋子编号[-7,7]
    Returns:
        int: 从0开始的序号
    """
    assert x != 0, "要求非空棋子"
    if x > 0:
        return x + NUM_PIECE - 1
    elif x < 0:
        return -x - 1


def get_decode_piece(x):
    if x >= 7:
        return x - NUM_PIECE + 1
    return -(x + 1)


def ps2feature(ps):
    """棋子特征编码
    Args:
        ps (np.array): 棋子[10,9]
    Returns:
        3d: (14,10,9) array
    """
    f = np.zeros((2 * NUM_PIECE, NUM_ROW, NUM_COL))
    # 14 * 10 * 9 【忽略0值】
    arr = ps.reshape((NUM_ROW, NUM_COL))
    idx = np.nonzero(arr)
    for y, x in zip(idx[0], idx[1]):
        pid = arr[y, x]
        i = get_encoded_piece(pid)
        f[i, NUM_ROW - y - 1, x] = 1
    return f


def encoded_action(action: int, lr: bool = False):
    """编码移动序号

    Args:
        action (int): 移动序号

    Returns:
        ndarray: np.ndarray(2,10,9)
    """
    res = np.zeros((2, NUM_ROW, NUM_COL), dtype=np.uint8)
    if action == NUM_ACTIONS:
        return res
    else:
        move = xqcpp.a2m(action)
        if lr:
            move = xqcpp.move2lr(move)
        x0, y0, x1, y1 = move_to_coordinate(move, True)
        res[0][y0][x0] = 1
        res[1][y1][x1] = 1
        return res


# def concatenated_feature(
#     state,
#     steps,
#     e,
#     p,
#     lr: bool = False,
# ):
#     """棋局合并特征
#     Args:
#         obs_dict (dict): 棋局特征字典
#         lr (bool, optional): 是否左右互换. Defaults to False.
#     Returns:
#         3d: (17, 10, 9) ndarray
#     """
#     # 6480 bytes
#     # 100万 ~6G
#     res = np.concatenate(
#         (
#             state if not lr else np.fliplr(state),
#             np.full((1, NUM_ROW, NUM_COL), steps) / MAX_EPISODE_STEPS,
#             np.full((1, NUM_ROW, NUM_COL), e) / MAX_NUM_NO_EAT,
#             np.full((1, NUM_ROW, NUM_COL), p) / NUM_PLAYER,
#         ),
#         axis=0,
#         dtype="float32",
#     )
#     return res


def concatenated_feature(
    state,
    p,
    last_action: int,
    lr: bool = False,
):
    """棋局合并特征
    Args:
        obs_dict (dict): 棋局特征字典
        lr (bool, optional): 是否左右互换. Defaults to False.
    Returns:
        3d: (17, 10, 9) ndarray
    """
    # 6480 bytes
    # 100万 ~6G
    res = np.concatenate(
        (
            state if not lr else np.fliplr(state),
            encoded_action(last_action, lr),
            # np.full((1, NUM_ROW, NUM_COL), steps) / MAX_EPISODE_STEPS,
            # np.full((1, NUM_ROW, NUM_COL), e) / MAX_NUM_NO_EAT,
            np.full((1, NUM_ROW, NUM_COL), p) / NUM_PLAYER,
        ),
        axis=0,
        dtype="float32",
    )
    return res


def obs2feature(obs_dict: dict, info: dict, lr: bool = False, flatten: bool = True):
    """棋局特征
    Args:
        obs_dict (dict): 棋局特征字典
        info (dict): 信息字典
        lr (bool, optional): 是否左右互换. Defaults to False.
        flatten (bool, optional): 是否展开为2维数组. Defaults to True.
    Returns:
        ndarray: 4d (n,PLANE_NUM, NUM_ROW, NUM_COL) ndarray or 2d (n,PLANE_NUM*NUM_ROW*NUM_COL)
    """
    is_vec = hasattr(obs_dict["to_play"], "__len__")
    n = 1
    if is_vec:
        n = len(obs_dict["to_play"])
    res = np.zeros((n, PLANE_NUM, NUM_ROW, NUM_COL), dtype="float32")
    if not is_vec:
        state = ps2feature(obs_dict["s"])
        # steps = obs_dict["steps"]
        # e = obs_dict["continuous_uneaten"]
        p = obs_dict["to_play"]
        last_action = info["last_action"]
        # res[0] = concatenated_feature(state, steps, e, p, lr)
        res[0] = concatenated_feature(state, p, last_action, lr)
    else:
        for i in range(n):
            state = ps2feature(obs_dict["s"][i])
            # steps = obs_dict["steps"][i]
            # e = obs_dict["continuous_uneaten"][i]
            p = obs_dict["to_play"][i]
            last_action = info["last_action"][i]
            # res[i] = concatenated_feature(state, steps, e, p, lr)
            res[i] = concatenated_feature(state, p, last_action, lr)
    if flatten:
        return res.reshape((n, -1))
    else:
        return res


def _one_to_coordinate(plane):
    # one-hot plane -> coordinate
    assert plane.shape == (NUM_ROW, NUM_COL)
    s = np.nonzero(plane)
    y, x = s[0].item(), s[1].item()
    return x, NUM_ROW - y - 1


def decode_action(encoded: np.ndarray, to_move=False):
    """编码移动特征解码
    Args:
        encoded (np.ndarray): 特征值
        to_move (bool, optional): 转换为移动字符串. Defaults to False.
    Returns:
        str | int: 移动字符串或移动序号
    """
    assert isinstance(encoded, np.ndarray), "解码对象必须为ndarray"
    expected_shape = (NUM_PLAYER, NUM_ROW, NUM_COL)
    assert encoded.shape == expected_shape, "期望shape = {}, 输入 = {}".format(
        expected_shape, encoded.shape
    )
    # 空白
    if np.count_nonzero(encoded) == 0:
        if to_move:
            return "0000"
        return NUM_ACTIONS
    x0, y0 = _one_to_coordinate(encoded[0])
    x1, y1 = _one_to_coordinate(encoded[1])
    move = "{}{}{}{}".format(x0, y0, x1, y1)
    if to_move:
        return move
    else:
        return xqcpp.m2a(move)


def decode_observation(observation: np.ndarray):
    """观察对象特征解码
    Args:
        observation (np.ndarray): 观察特征【棋子、连续未吃子、to play、最后移动】
    Returns:
        dict: 解码字典
    """
    assert isinstance(observation, np.ndarray), "解码对象必须为ndarray"
    expected_shape = (PLANE_NUM, NUM_ROW, NUM_COL)
    assert observation.shape == expected_shape, "期望shape = {}, 输入 = {}".format(
        expected_shape, observation.shape
    )
    ps_arr = np.zeros((NUM_ROW, NUM_COL), dtype=np.int8)
    n = NUM_PLAYER * NUM_PIECE
    ps = observation[:n]
    for i in range(n):
        pid = get_decode_piece(i)
        plane = ps[i]
        idx = np.nonzero(plane)
        ps_arr[idx] = pid
    continuous_uneaten = observation[n][0][0] * MAX_NUM_NO_EAT
    to_play = observation[n + 1][0][0] * NUM_PLAYER
    last_a = decode_action(observation[n + 2 :])
    return {
        "s": np.flipud(ps_arr),
        "continuous_uneaten": int(continuous_uneaten),
        "to_play": int(to_play),
        "last_a": int(last_a),
    }


def to_fen_str(pieces: np.ndarray, to_play: int = 1, steps: int = 1, no_eat: int = 0):
    """将二维棋子编码转换为fen字符串
    Args:
        pieces (np.ndarray): 棋子编码表
        to_play (int): 下一步走子方 【1 代表红方 2 代表黑方】
        steps (int): 步数
        no_eat (int): 连续未吃子数量
    Returns:
        str: fen字符串
    """
    assert isinstance(pieces, np.ndarray), "解码对象必须为ndarray"
    expected_shape = (NUM_ROW, NUM_COL)
    assert pieces.max() == 7
    assert pieces.min() == -7
    assert np.count_nonzero(pieces) <= 32
    fen = ""
    for y in range(NUM_ROW):
        e = 0
        row = ""
        for x in range(NUM_COL):
            ps = PIECE_MAPS[pieces[9 - y, x]]
            if ps == "+":
                e += 1
            else:
                if e > 0:
                    row += str(e)
                    e = 0
                row += ps
        if e > 0:
            row += str(e)
        fen += row
        if y != 9:
            fen += "/"
    half = steps // 2 + steps % 2
    return "{} {} - {} {} {}".format(
        fen, "r" if to_play == 1 else "b", no_eat, half, steps
    )


def show_board_info(observation: np.ndarray):
    """解码演示棋盘
    Args:
        observation (np.ndarray): 棋盘编码信息
    """
    d = decode_observation(observation)
    fen = to_fen_str(d["s"], d["to_play"], 1, d["continuous_uneaten"])
    board = xqcpp.XqBoard()
    board.reset()
    board.set_use_rule_flag(False)
    board.init_set(fen, True, True)
    print(render_board_to_text(board, "演示棋盘"))
    if d["last_a"] != NUM_ACTIONS:
        print("上一步:{}".format(xqcpp.m2a(d["last_a"])))
    print("轮到{}方走子".format("红" if d["to_play"] == 1 else "黑"))


def show_board_by_pieces(pieces, to_play=RED_PLAYER):
    """演示棋盘

    Args:
        pieces (ndarray): 2d (10,9)
    """
    board = xqcpp.XqBoard()
    board.reset()
    fen = to_fen_str(pieces)
    board.init_set(fen, True, True)
    player = "红方" if to_play == RED_PLAYER else "黑方"
    board.set_player(player)
    print(render_board_to_text(board, "演示棋盘"))
