import math

import numpy as np


def set_one_thread():
    import os
    import torch

    # import numpy as np

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def set_seed_all(seed):
    import torch
    import numpy as np
    import random
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)


def cal_scale_reward(val, normalizer, exp=False, beta=1.0):
    # import math

    # reward = 0
    if not exp:
        reward = min(val / normalizer, 10)
    else:
        reward = (
            val / normalizer
            if val <= normalizer
            else (math.exp(beta * (min(val / normalizer, 2))) - 1) / beta
        )
    return reward


def normalize_tables(tables, max_val, min_val):
    t_max_val = np.max(tables)
    t_min_val = np.min(tables)

    return (tables - t_min_val) * (max_val - min_val) / (
        t_max_val - t_min_val
    ) + min_val


# import numpy as np
#
# payoff_tables = np.random.random([2, 4, 4])
# print(payoff_tables)
#
# print(normalize_tables(tables=payoff_tables, max_val=10, min_val=-10))
