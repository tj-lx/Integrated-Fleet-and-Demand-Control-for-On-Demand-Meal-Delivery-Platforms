import torch
import torch.nn as nn
import config

class QNetwork(nn.Module):
    def __init__(self, num_restaurants, embedding_dim):
        super(QNetwork, self).__init__()
        
        # [cite_start]1. 特征编码层 (Stop Embedding) [cite: 568]
        # 输入: [位置x, 位置y, 类型(0/1), 预估时间]
        self.stop_encoder = nn.Linear(4, embedding_dim)
        
        # [cite_start]2. 路线注意力机制 (Route Attention) [cite: 569]
        # 将变长的路线压缩成固定向量
        self.route_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        
        # [cite_start]3. 车队注意力机制 (Fleet Attention) [cite: 571]
        # 将变长的车队(多个路线)压缩成全局状态向量
        self.fleet_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        
        # [cite_start]4. Dueling Head [cite: 577]
        # Value Stream (State Value)
        self.value_stream = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage Stream (Action Value for each Restaurant-Vehicle pair)
        # 输入: State Embedding + Restaurant Embedding
        self.rest_embedding = nn.Embedding(num_restaurants, embedding_dim)
        self.advantage_stream = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64), # Fleet State + Rest Feature
            nn.ReLU(),
            nn.Linear(64, config.NUM_VEHICLES) # 输出每个车辆的Q值
        )

    def forward(self, fleet_states, restaurant_ids):
        """
        fleet_states: List of tensors, shape (Batch, Num_Vehicles, Max_Stops, 4)
        restaurant_ids: (Batch, )
        """
        batch_size = restaurant_ids.size(0)
        
        # --- Step 1 & 2: Route Embedding ---
        # 这里为了简化，假设输入已经是Padding好的Tensor
        # 实际工程中需要处理变长序列 mask
        # fleet_states: (B, N_Veh, N_Stops, 4)
        
        # 合并Batch和Vehicle维度进行处理
        B, N, S, F = fleet_states.shape
        x = fleet_states.view(B * N, S, F) 
        
        x = torch.relu(self.stop_encoder(x)) # (B*N, S, Emb)
        
        # Route Attention: Query=Key=Value=Stops
        # 输出: (B*N, S, Emb) -> 取平均或池化得到 (B*N, Emb)
        attn_out, _ = self.route_attention(x, x, x)
        route_embeddings = attn_out.mean(dim=1) # (B*N, Emb)
        
        # --- Step 3: Fleet Embedding ---
        # 恢复维度 (B, N, Emb)
        fleet_input = route_embeddings.view(B, N, -1)
        
        # Fleet Attention
        fleet_out, _ = self.fleet_attention(fleet_input, fleet_input, fleet_input)
        state_embedding = fleet_out.mean(dim=1) # (B, Emb) 全局状态
        
        # --- Step 4: Dueling Heads ---
        # V(s)
        V = self.value_stream(state_embedding) # (B, 1)
        
        # A(s, a)
        # 获取当前正在决策的餐厅的Embedding
        r_embed = self.rest_embedding(restaurant_ids) # (B, Emb)
        
        # 将状态和餐厅特征结合
        combined = torch.cat([state_embedding, r_embed], dim=1) # (B, Emb*2)
        A = self.advantage_stream(combined) # (B, Num_Vehicles)
        
        # Q(s, r, v) = V(s) + A(s, r, v) - mean(A)
        Q = V + A - A.mean(dim=1, keepdim=True)
        
        return Q