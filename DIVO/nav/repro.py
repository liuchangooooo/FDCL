"""复现与日志工具(Req 13.4/13.5)。"""
import os
import random

import numpy as np


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# 训练/评测应记录的关键字段(train_skill 已记 val 指标;完整字段见此)
LOG_FIELDS = [
    "step", "reward_task", "reward_total",          # 严格区分任务与多样增广回报
    "K_eff", "action_var", "sat_rate",
    "boundary_count", "boundary_rate", "mean_b",
    "r_hard", "r_easy", "val_test_mean_score",
]
