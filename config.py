import torch

# 环境设置
GRID_SIZE = 10         # 10km x 10km 区域
NUM_RESTAURANTS = 5    # 论文中是110，我们缩小规模
NUM_VEHICLES = 3       # 论文中可以是100，我们用3个
SERVICE_TIME_LIMIT = 40 # 40分钟承诺送达时间
VELOCITY = 0.5         # 车速 0.5 km/min (30km/h)

# 训练超参数
BATCH_SIZE = 32
LR = 1e-4
GAMMA = 0.99           # 折扣因子
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 1000
MEMORY_SIZE = 10000
NUM_EPISODES = 500     # 训练轮数

# 神经网络维度
EMBEDDING_DIM = 32     # 论文中可能更大，这里轻量化

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")