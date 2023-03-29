import os


def get_experiment_path(runs: int = 0):
    """记录N次实验结果根目录
    Args:
        runs (int, optional): 实验次数. Defaults to 0.
    Returns:
        str: path
    """
    root = os.path.expanduser("~")
    path = os.path.join(root, "muzero_experiment", "runs_{:03d}".format(runs))
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_latest_run_path():
    """获取最新检查点根目录
    Returns:
        str: path
    """
    exp_path = get_experiment_path()
    parent = os.path.dirname(exp_path)
    # List all the checkpoints.
    checkpoint_ids = []
    for file in os.listdir(parent):
        if file[:5] == "runs_":
            checkpoint_id = file.split(".")[0]
            checkpoint_ids.append(int(checkpoint_id[5:]))
    checkpoint_id = max(checkpoint_ids)
    return os.path.join(parent, f"runs_{checkpoint_id:03d}")
