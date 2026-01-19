from src.environment import DeliveryEnvironment
from src.agent import Agent
import config
import torch
import random

def main():
    env = DeliveryEnvironment()
    agent = Agent()
    
    print(f"Start training on synthetic data for {config.NUM_EPISODES} episodes...")
    
    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 1. 获得 Q 值矩阵 (Num_Rest, Num_Veh)
            # 探索 (Epsilon-Greedy): 偶尔随机生成 Q 值矩阵以模拟随机决策
            if random.random() < agent.epsilon:
                q_values = torch.rand(config.NUM_RESTAURANTS, config.NUM_VEHICLES).to(config.DEVICE)
            else:
                q_values = agent.get_action_values(state)
            
            # 2. 环境执行 (Environment Step)
            # 环境内部包含了 Demand Control (排序) 和 Fleet Control (派单)
            next_state, reward, done = env.step(q_values)
            
            # 3. 存储经验 (简化版，实际需要存储具体的 action index)
            # agent.memory.append((state, q_values, reward, next_state, done))
            
            # 4. 训练
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
        # 衰减 Epsilon
        agent.epsilon = max(config.EPSILON_END, agent.epsilon * 0.99)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward (Negative Delay): {total_reward:.2f}")
            
    # 保存模型
    torch.save(agent.policy_net.state_dict(), "results/models/integrated_model.pth")
    print("Training finished.")

if __name__ == "__main__":
    main()