import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# === config.py ===
NUM_RESTAURANTS = 3
NUM_VEHICLES = 3
GRID_SIZE = 10.0
SPEED = 1.0
PROMISED_TIME = 40.0
FEATURE_DIM = 6   # Stop 特征维度 (类型, x, y, 剩余时间, etc.)
HIDDEN_DIM = 32   # 神经网络隐藏层维度
BATCH_SIZE = 16
LR = 1e-3
GAMMA = 0.99
MAX_STOPS = 10    # 限制每辆车最大任务数 (为了Padding方便，简化运行)

RESTAURANTS = [
    {'id': 0, 'loc': (2.0, 2.0), 'base_u': 1.0},
    {'id': 1, 'loc': (8.0, 8.0), 'base_u': 0.8},
    {'id': 2, 'loc': (2.0, 8.0), 'base_u': 0.9}
]