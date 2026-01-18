import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class IntegratedAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Stop Embedding Layer (把单个站点的特征变成向量)
        self.stop_embed = nn.Linear(config.FEATURE_DIM, config.HIDDEN_DIM)
        
        # 2. Route Attention Mechanism (聚合一辆车的所有站点)
        # Query是车辆本身特征(简化为固定向量), Key/Value是Stop Embeddings
        self.route_attn = nn.MultiheadAttention(config.HIDDEN_DIM, num_heads=4, batch_first=True)
        self.route_query = nn.Parameter(torch.randn(1, 1, config.HIDDEN_DIM)) # Learnable Query
        
        # 3. Fleet Attention Mechanism (聚合所有车辆)
        self.fleet_attn = nn.MultiheadAttention(config.HIDDEN_DIM, num_heads=4, batch_first=True)
        self.fleet_query = nn.Parameter(torch.randn(1, 1, config.HIDDEN_DIM))
        
        # 4. Context Embedding (当前客户和餐厅的特征)
        self.context_embed = nn.Linear(4, config.HIDDEN_DIM) # [req_x, req_y, rest_x, rest_y]
        
        # 5. Dueling Heads (Value & Advantage)
        # 输入: Fleet State + Context + Specific Vehicle Route
        combined_dim = config.HIDDEN_DIM * 3 # Fleet + Route + Context
        
        self.val_fc = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.adv_fc = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, vehicles_stops, vehicles_mask, context):
        """
        vehicles_stops: [Batch, Num_Vehicles, Max_Stops, Feat_Dim]
        vehicles_mask: [Batch, Num_Vehicles, Max_Stops] (1 for padding, 0 for data)
        context: [Batch, 4] (当前处理的 Request + Restaurant 特征)
        """
        B, N_V, MAX_S, F_D = vehicles_stops.shape
        
        # === A. Route Attention (并行处理所有车辆) ===
        # Reshape to [B * N_V, MAX_S, F_D]
        stops_flat = vehicles_stops.view(B * N_V, MAX_S, F_D)
        mask_flat = vehicles_mask.view(B * N_V, MAX_S)
        
        # Embedding
        x_stops = F.relu(self.stop_embed(stops_flat)) # [B*N_V, MAX_S, Hidden]
        
        # Attention Pooling -> Route Embeddings
        # Query 广播到每个车辆
        q_route = self.route_query.expand(B * N_V, 1, config.HIDDEN_DIM)
        # key_padding_mask 需要 Bool Tensor (True 表示忽略)
        route_emb, _ = self.route_attn(q_route, x_stops, x_stops, key_padding_mask=mask_flat.bool())
        route_emb = route_emb.squeeze(1).view(B, N_V, config.HIDDEN_DIM) # [B, N_V, Hidden]
        
        # === B. Fleet Attention (聚合车队信息) ===
        q_fleet = self.fleet_query.expand(B, 1, config.HIDDEN_DIM)
        fleet_emb, _ = self.fleet_attn(q_fleet, route_emb, route_emb) # [B, 1, Hidden]
        fleet_vec = fleet_emb.repeat(1, N_V, 1) # 广播回每个车辆用于计算Advantage [B, N_V, Hidden]
        
        # === C. Context Fusion ===
        ctx_vec = F.relu(self.context_embed(context)).unsqueeze(1).expand(B, N_V, config.HIDDEN_DIM)
        
        # === D. Dueling Network Calculation ===
        # 我们需要评估每一辆车 v 的 Q值
        # Input: [Fleet_Summary, Vehicle_v_Route_Summary, Context]
        combined = torch.cat([fleet_vec, route_emb, ctx_vec], dim=-1) # [B, N_V, Hidden*3]
        
        values = self.val_fc(combined) # State Value V(S)
        advantages = self.adv_fc(combined) # Advantage A(S, v)
        
        # Q(S, v) = V(S) + A(S, v) - mean(A(S, .))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values.squeeze(-1) # [B, N_V]