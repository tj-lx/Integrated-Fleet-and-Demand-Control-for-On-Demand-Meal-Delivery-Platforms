import torch
import torch.optim as optim
import random
import config
from src.network import QNetwork

class Agent:
    def __init__(self):
        self.policy_net = QNetwork(config.NUM_RESTAURANTS, config.EMBEDDING_DIM).to(config.DEVICE)
        self.target_net = QNetwork(config.NUM_RESTAURANTS, config.EMBEDDING_DIM).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        self.memory = []
        self.epsilon = config.EPSILON_START

    def get_action_values(self, state):
        # 输出所有 (Restaurant, Vehicle) 的 Q 值矩阵
        # 实际前向传播需要对每个Restaurant跑一次，或者Batch并行
        # 这里用Batch并行技巧
        state = state.repeat(config.NUM_RESTAURANTS, 1, 1, 1) # (Num_Rest, N_Veh, S, F)
        rest_ids = torch.arange(config.NUM_RESTAURANTS).to(config.DEVICE)
        
        with torch.no_grad():
            q_values = self.policy_net(state, rest_ids) # (Num_Rest, Num_Vehicles)
        return q_values

    def train_step(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        
        # 简单的 Experience Replay 采样
        batch = random.sample(self.memory, config.BATCH_SIZE)
        # ... (标准的 DQN Loss 计算代码，这里省略具体 tensor 拼接细节以节省篇幅)
        # 关键点：Loss = (Q(s, chosen_rest, chosen_veh) - (r + gamma * max Q(s', ...)))^2