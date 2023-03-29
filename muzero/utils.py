import datetime
import math

import numpy as np
import xqcpp
from gymxq.constants import NUM_ACTIONS, NUM_COL, NUM_ROW


def duration_repr(duration: float):
    """用时表示

    Args:
        duration (float): 时间差

    Returns:
        str: 用时字符串【x days, xx:xx:xx】
    """
    rep = datetime.timedelta(seconds=duration)
    return str(rep)[:-7]


def get_lr_policy(policy):
    """左右移动互换后的政策

    Args:
        policy (list or dict): 政策

    Raises:
        TypeError: 触发异常

    Returns:
        list or dict: 概率不变，移动左右互换
    """
    msg = "概率之和应接近1或者为空"
    if isinstance(policy, list):
        v = math.ceil(round(sum(policy), 2))
        assert v == 0 or v == 1, msg
        res = [0] * NUM_ACTIONS
        for a, p in enumerate(policy):
            if p > 0.0:
                m = xqcpp.a2m(a)
                lr_a = xqcpp.m2a(xqcpp.move2lr(m))
                res[lr_a] = p
        return res
    elif isinstance(policy, dict):
        v = math.ceil(round(sum(policy.values()), 2))
        assert round(v, 2) == 0 or round(v, 2) == 1, msg
        func_a = lambda x: xqcpp.m2a(xqcpp.move2lr(xqcpp.a2m(x)))
        func_m = lambda x: xqcpp.move2lr(x)
        func = func_a if isinstance(list(policy.keys())[0], int) else func_m
        res = {}
        for k, v in policy.items():
            res[func(k)] = v
        return res
    else:
        raise TypeError("policy要么为列表，要么为字典，不支持类型{}".format(type(policy)))


def get_lr_feature(feature):
    """左右互换后的特征

    Args:
        feature (ndarray): >= 2d

    Returns:
        ndarray: 左右互换后的特征
    """
    assert feature.ndim >= 2
    assert feature.shape[-2:] == (NUM_ROW, NUM_COL)
    s = feature.shape
    n = len(s)
    return np.flip(feature, n - 1)
