import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from src.utils import extract_features  # 确保你创建了 utils.py

# === 1. 神经网络定义 (原 network.py 的内容) ===

class IntegratedAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stop Embedding: 将站点特征映射到隐藏层
        self.stop_embed = nn.Linear(config.FEATURE_DIM, config.HIDDEN_DIM)
        
        # Route Attention: 聚合一辆车的所有站点
        self.route_attn = nn.MultiheadAttention(config.HIDDEN_DIM, num_heads=4, batch_first=True)
        self.route_query = nn.Parameter(torch.randn(1, 1, config.HIDDEN_DIM))
        
        # Fleet Attention: 聚合所有车辆
        self.fleet_attn = nn.MultiheadAttention(config.HIDDEN_DIM, num_heads=4, batch_first=True)
        self.fleet_query = nn.Parameter(torch.randn(1, 1, config.HIDDEN_DIM))
        
        # Context Embedding: 当前订单和餐厅的特征
        self.context_embed = nn.Linear(4, config.HIDDEN_DIM)
        
        # Dueling Heads: 计算 Value 和 Advantage
        combined_dim = config.HIDDEN_DIM * 3
        self.val_fc = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.adv_fc = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, vehicles_stops, vehicles_mask, context):
        B, N_V, MAX_S, F_D = vehicles_stops.shape
        
        # 1. Route Attention
        stops_flat = vehicles_stops.view(B * N_V, MAX_S, F_D)
        mask_flat = vehicles_mask.view(B * N_V, MAX_S)
        
        x_stops = F.relu(self.stop_embed(stops_flat))
        
        q_route = self.route_query.expand(B * N_V, 1, config.HIDDEN_DIM)
        # 注意：mask需要是 Bool 类型 (True表示Padding)
        route_emb, _ = self.route_attn(q_route, x_stops, x_stops, key_padding_mask=mask_flat.bool())
        route_emb = route_emb.squeeze(1).view(B, N_V, config.HIDDEN_DIM)
        
        # 2. Fleet Attention
        q_fleet = self.fleet_query.expand(B, 1, config.HIDDEN_DIM)
        fleet_emb, _ = self.fleet_attn(q_fleet, route_emb, route_emb)
        fleet_vec = fleet_emb.repeat(1, N_V, 1)
        
        # 3. Context Fusion
        ctx_vec = F.relu(self.context_embed(context)).unsqueeze(1).expand(B, N_V, config.HIDDEN_DIM)
        
        # 4. Combine & Output
        combined = torch.cat([fleet_vec, route_emb, ctx_vec], dim=-1)
        
        values = self.val_fc(combined)
        advantages = self.adv_fc(combined)
        
        # Dueling DQN aggregation
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values.squeeze(-1) # Output: [B, N_V]

# === 2. 决策逻辑 ===

def make_integrated_decision(model, env):
    """
    集成决策函数
    1. 遍历所有餐厅
    2. 用模型预测每个餐厅派给每辆车的 Q 值
    3. 选出每个餐厅的最佳车辆 (Fleet Control)
    4. 根据最佳 Q 值对餐厅排序 (Demand Control)
    """
    model.eval() # 预测模式
    
    best_vehicles = []   # 存储: 餐厅 r -> 最佳车辆 v
    min_q_per_rest = []  # 存储: 餐厅 r -> 最小延误 Q
    
    # 遍历每个餐厅进行评估
    for r_idx in range(config.NUM_RESTAURANTS):
        # 提取特征 (调用 utils.py)
        stops, mask, ctx = extract_features(env, r_idx)
        
        with torch.no_grad():
            # 预测: [1, Num_Vehicles]
            q_vals = model(stops, mask, ctx)
        
        # Fleet Control: 选出 Q 值最小（延误最小）的车
        best_v = torch.argmin(q_vals).item()
        min_q = q_vals[0, best_v].item()
        
        best_vehicles.append(best_v)
        min_q_per_rest.append(min_q)
    
    # Demand Control: 根据预测的延误从小到大排序
    # argsort 返回的是索引，比如 [2, 0, 1] 表示餐厅2延误最小，排第一
    ranking_indices = np.argsort(min_q_per_rest)
    
    return ranking_indices, best_vehicles