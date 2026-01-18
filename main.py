import torch
import torch.optim as optim
import numpy as np
import config
from src.environment import DeliveryEnv
from src.network import IntegratedAttentionModel
from src.utils import extract_features

def main():
    env = DeliveryEnv()
    model = IntegratedAttentionModel()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    # 模拟 MNL 选择
    def user_choice(ranking):
        utilities = [np.exp(r['base_u'] + rank * -0.5) for rank, r in enumerate(ranking)]
        probs = np.array(utilities) / sum(utilities)
        return np.random.choice(len(ranking), p=probs)

    print("开始高保真训练 (Attention + Routing Logic)...")
    
    for episode in range(200): # 简化的 Loop
        env.reset()
        total_loss = 0
        
        for step in range(20): # 每个 Episode 20 步
            # === 1. Integrated Decision Making ===
            # 我们需要评估每个餐厅 r 的最佳车辆 v 的 Q值
            
            best_vehicles = [] # map: r_idx -> v_idx
            min_q_per_rest = [] # map: r_idx -> min_q
            
            # 这一步可以 Batch 化处理所有餐厅，为了逻辑清晰先用 Loop
            for r_idx in range(config.NUM_RESTAURANTS):
                stops, mask, ctx = extract_features(env, r_idx)
                
                # 前向传播：得到这一个餐厅，分配给所有车辆的 Q值 [1, Num_Vehicles]
                with torch.no_grad():
                    q_vals = model(stops, mask, ctx) 
                
                # Fleet Control: 选 Q 最小的车
                best_v = torch.argmin(q_vals).item()
                min_q = q_vals[0, best_v].item()
                
                best_vehicles.append(best_v)
                min_q_per_rest.append(min_q)
            
            # Demand Control: 按 min_q 排序 (Q代表延误，越小越好)
            # 得到 ranking: 比如 [2, 0, 1] 表示餐厅2排第一
            ranking_indices = np.argsort(min_q_per_rest)
            ranking_data = [config.RESTAURANTS[i] for i in ranking_indices]
            
            # === 2. Interaction & Transition ===
            choice_rank = user_choice(ranking_data) # 用户基于排序选了第几个
            chosen_r_idx = ranking_indices[choice_rank] # 对应的真实餐厅ID
            chosen_v_idx = best_vehicles[chosen_r_idx] # 对应的预案车辆
            
            # 真正的环境交互：物理插入
            reward = env.step(chosen_r_idx, chosen_v_idx)
            
            # === 3. Learning (简化版 Q-Learning) ===
            # 真正的 Q-Learning 应该用 Replay Buffer 和 Target Net
            # 这里演示 loss 计算逻辑：Target = Reward + Gamma * min(Q_next)
            
            # Re-calculate current Q (with grad)
            stops, mask, ctx = extract_features(env, chosen_r_idx)
            current_q_all = model(stops, mask, ctx)
            current_q = current_q_all[0, chosen_v_idx]
            
            # 简化的 TD Target (假设 Next Q 为 0，仅拟合当前 Reward)
            # 为了完整复现，这里应该再算一遍 Next State 的 Max Q
            target = torch.tensor([-reward], dtype=torch.float32) # Minimize Delay
            
            loss = F.mse_loss(current_q, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if episode % 10 == 0:
            print(f"Episode {episode}, Avg Loss: {total_loss/20:.4f}")

if __name__ == "__main__":
    main()