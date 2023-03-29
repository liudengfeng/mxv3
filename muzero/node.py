import os

import graphviz
import numpy as np
import torch
import xqcpp
from gymxq.constants import BLACK_PLAYER, RED_PLAYER


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        # 子节点关键字为整数
        self.children = {}
        self.hidden_state = None
        self.reward = 0

        self.policy = {}
        # for graphviz
        self.parent = None
        # self.depth_ = 0
        self.path_ = "root"
        # 注意，此处为字符串，代表移动字符串
        self.action = "root"
        self.ucb_score = 0

    def __hash__(self) -> int:
        return hash(self.path_)

    def __eq__(self, other):
        return self.path() == other.path()

    def path(self):
        return self.path_

    def depth(self):
        return len(self.path_.split(" -> ")) - 1

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(
        self,
        actions,
        to_play,
        reward,
        policy_logits,
        hidden_state,
        use_policy=False,
        debug=False,
    ):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        if not use_policy:
            policy_values = torch.softmax(
                torch.tensor([policy_logits[0][a] for a in actions]), dim=0
            ).tolist()
            policy = {a: policy_values[i] for i, a in enumerate(actions)}
        else:
            # 以下测试专用
            assert isinstance(policy_logits, dict), "policy_logits应为字典对象"
            assert all(
                [isinstance(k, int) for k in policy_logits.keys()]
            ), "测试时所提供的政策，其关键字应为代表移动的整数编码"
            policy = policy_logits

        self.policy = policy
        for action, p in policy.items():
            # 添加概率限制
            if p > 0.0:
                c = Node(p)
                c.to_play = BLACK_PLAYER if self.to_play == RED_PLAYER else RED_PLAYER
                c.action = xqcpp.a2m(action)
                c.path_ = "{} -> {}".format(self.path(), c.action)
                self.children[action] = c
        if debug:
            # 调试模式需要与父节点链接
            for c in self.children.values():
                c.parent = self

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

    def get_updated_policy(self):
        # 更新后的政策(用于显示，精确到2位)
        sum_visits = sum(child.visit_count for child in self.children.values())
        if sum_visits > 0:
            return {
                xqcpp.a2m(a): round(child.visit_count / sum_visits, 2)
                for a, child in self.children.items()
            }
        else:
            return {}

    def get_root_value(self):
        if self.parent is None:
            return max([c.value() for c in self.children.values()])
        return self.value()


def get_root_node_table_like_label(state: Node):
    """根节点label
    Args:
        state (Node): 根节点
    Returns:
        str: 节点标签
    """
    return """<
<TABLE>
  <TR>
    <TD ALIGN='LEFT'>State</TD>
    <TD ALIGN='RIGHT'>Root</TD>
  </TR>
    <TR>
    <TD ALIGN='LEFT'>Vsum</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Visit</TD>
    <TD ALIGN='RIGHT'>{:4d}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Value</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
</TABLE>>""".format(
        state.value_sum,
        state.visit_count,
        state.value(),
    )


def get_node_table_like_label(state: Node):
    return """<
<TABLE>
  <TR>
    <TD ALIGN='LEFT'>Reward</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Vsum</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Value</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>UCB</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Prior</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Visit</TD>
    <TD ALIGN='RIGHT'>{:4d}</TD>
  </TR>
</TABLE>>""".format(
        state.reward,
        state.value_sum,
        state.value(),
        state.ucb_score,
        state.prior,
        state.visit_count,
    )


def default_node_decorator(state: Node):
    """Decorates a state-node of the game tree.
    This method can be called by a custom decorator to prepopulate the attributes
    dictionary. Then only relevant attributes need to be changed, or added.
    Args:
      state: The state.
    Returns:
      `dict` with graphviz node style attributes.
    """
    _PLAYER_COLORS = {-1: "black", 1: "red", 2: "blue"}
    player = state.parent.to_play if state.parent else -1
    attrs = {}
    attrs["label"] = get_node_table_like_label(state)
    attrs["color"] = _PLAYER_COLORS.get(player, "black")
    return attrs


def default_edge_decorator(child: Node):
    """Decorates a state-node of the game tree.
    This method can be called by a custom decorator to prepopulate the attributes
    dictionary. Then only relevant attributes need to be changed, or added.
    Args:
      parent: The parent state.
    Returns:
      `dict` with graphviz node style attributes.
    """
    _PLAYER_COLORS = {-1: "black", 1: "red", 2: "blue"}
    player = child.parent.to_play if child.parent else -1
    attrs = {}
    attrs["label"] = child.action
    attrs["color"] = _PLAYER_COLORS.get(player, "black")
    return attrs


def build_mcts_tree(dot, state, depth):
    """Recursively builds the mcts tree."""
    if not state.expanded():
        return

    for child in state.children.values():
        if child.visit_count >= 1:
            dot.node(child.path(), **default_node_decorator(child))
            dot.edge(
                child.parent.path(),
                child.path(),
                **default_edge_decorator(child),
            )
            build_mcts_tree(dot, child, depth + 1)


def render_root(
    root: Node,
    filename: str,
    format: str = "png",
    saved_path=None,
):
    """演示根节点
    Args:
        root (Node): 根节点
        filename (str):  文件名称
        format (str, optional): 输出文件格式. Defaults to "png".
        saved_path (str, optional):  存储路径
    """
    assert format in ["png", "svg"], "仅支持png和svg格式"
    graph_attr = {"rankdir": "LR", "fontsize": "8"}
    node_attr = {"shape": "plaintext"}
    # 不需要扩展名
    name = filename.split(".")[0]
    dot = graphviz.Digraph(
        name,
        comment="蒙特卡洛搜索树",
        format=format,
        graph_attr=graph_attr,
        node_attr=node_attr,
    )
    dot.node("root", label=get_root_node_table_like_label(root), shape="oval")
    build_mcts_tree(dot, root, 0)
    # 尚未展开，to_play = -1
    # 多进程操作
    if saved_path:
        fp = os.path.join(saved_path, "mcts_{}".format(name))
    else:
        fp = "pid_{:06d}".format(name, os.getpid())
    dot.render(fp, view=False, cleanup=True)
