import random
import torch
import numpy as np
import config
from src.structures import *
from src.utils import get_distance, cheapest_insertion
from src.choice_model import ChoiceModel

class DeliveryEnvironment:
    def __init__(self):
        self.time = 0
        self.restaurants = [
            Restaurant(i, Location(random.uniform(0, config.GRID_SIZE), random.uniform(0, config.GRID_SIZE))) 
            for i in range(config.NUM_RESTAURANTS)
        ]
        self.vehicles = [
            Vehicle(i, Location(config.GRID_SIZE/2, config.GRID_SIZE/2)) 
            for i in range(config.NUM_VEHICLES)
        ]
        self.choice_model = ChoiceModel(config.NUM_RESTAURANTS)
        self.pending_orders = []
        self.order_count = 0

    def get_state_tensor(self):
        # 将当前车辆状态转换为Tensor供网络使用
        # 简化：Padding到固定长度 10 个stops
        max_stops = 10
        state_tensor = torch.zeros(1, config.NUM_VEHICLES, max_stops, 4)
        
        for i, v in enumerate(self.vehicles):
            for j, stop in enumerate(v.route[:max_stops]):
                state_tensor[0, i, j, 0] = stop.loc.x
                state_tensor[0, i, j, 1] = stop.loc.y
                state_tensor[0, i, j, 2] = 1.0 if stop.type == 'pickup' else 0.0
                state_tensor[0, i, j, 3] = stop.estimated_arrival - self.time
        return state_tensor.to(config.DEVICE)

    def step(self, action_q_values):
        """
        [cite_start]集成控制的核心逻辑 [cite: 467-471]
        action_q_values: 针对当前客户，每个(Restaurant, Vehicle)对的Q值矩阵
        """
        # 1. 需求控制 (Demand Control): 决定Display Configuration
        # 论文 Proposition 1: 最优Display取决于预期Q值
        # 这里为了简化，我们根据 Q_min (每个餐厅对应的最佳车辆的Q值) 对餐厅排序
        
        # 假设 action_q_values 形状为 (Num_Restaurants, Num_Vehicles)
        # 找到每个餐厅的最佳车辆指派带来的 Cost (Q值代表Delay，越小越好)
        best_vehicle_costs, best_vehicles = action_q_values.min(dim=1)
        
        # 排序餐厅：Cost越小，排越前面
        sorted_indices = torch.argsort(best_vehicle_costs)
        display_order = sorted_indices.tolist() # 这就是 \sigma_k
        
        # 2. 模拟用户选择 (Customer Choice)
        probs, ordered_rests = self.choice_model.get_choice_probabilities(self.restaurants, display_order)
        
        # 采样用户选择
        chosen_idx = torch.multinomial(probs, 1).item()
        chosen_restaurant = ordered_rests[chosen_idx]
        chosen_rest_original_idx = chosen_restaurant.id
        
        # 3. 车队控制 (Fleet Control)
        # 根据用户选的餐厅，指派给该餐厅对应的最佳车辆
        assigned_vehicle_idx = best_vehicles[chosen_rest_original_idx].item()
        
        # --- 执行物理仿真 ---
        self.order_count += 1
        new_customer_loc = Location(random.uniform(0, config.GRID_SIZE), random.uniform(0, config.GRID_SIZE))
        
        # 创建订单对象
        order = Order(self.order_count, chosen_rest_original_idx, new_customer_loc, self.time, self.time + 40)
        
        # 更新车辆路线 (使用 Cheapest Insertion)
        vehicle = self.vehicles[assigned_vehicle_idx]
        pickup = Stop(chosen_restaurant.loc, 'pickup', order.id)
        delivery = Stop(new_customer_loc, 'delivery', order.id)
        
        new_route, cost = cheapest_insertion(vehicle, pickup, delivery, self.time, config.VELOCITY)
        vehicle.route = new_route
        
        # 计算 Reward (负的 Delay)
        # Delay = max(0, Actual_Arrival - Promised)
        # 这里简单用 cost (完成时间) - Promised 近似
        delay = max(0, (self.time + cost) - order.promised_time)
        reward = -delay 
        
        # 时间推移 (模拟订单到达间隔)
        self.time += random.expovariate(1.0/5.0) # 平均5分钟一个订单
        
        done = self.order_count >= 50 # 一集50个订单
        
        return self.get_state_tensor(), reward, done
    
    def reset(self):
        self.__init__()
        return self.get_state_tensor()