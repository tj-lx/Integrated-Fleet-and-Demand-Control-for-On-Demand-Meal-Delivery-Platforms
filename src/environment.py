import numpy as np
import copy
import config
from src.structures import Stop, Vehicle

class DeliveryEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0.0
        self.vehicles = [Vehicle(i, (config.GRID_SIZE/2, config.GRID_SIZE/2)) for i in range(config.NUM_VEHICLES)]
        self.current_req = None
        self.req_id_counter = 0
        self.generate_request()
        return self

    def generate_request(self):
        self.req_id_counter += 1
        self.current_req = {
            'id': self.req_id_counter,
            'loc': (np.random.uniform(0, config.GRID_SIZE), np.random.uniform(0, config.GRID_SIZE)),
            'time': self.time
        }

    def get_travel_time(self, p1, p2):
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        return dist / config.SPEED

    def calculate_route_delay(self, vehicle: Vehicle, new_route: list):
        """计算一条路线的总延误 (Objective Function)"""
        current_time = max(self.time, vehicle.next_free_time)
        current_loc = vehicle.loc
        total_delay = 0.0
        
        for stop in new_route:
            travel = self.get_travel_time(current_loc, stop.loc)
            arrival = current_time + travel
            current_time = arrival
            current_loc = stop.loc
            
            # 只有 Delivery 节点才有延误惩罚
            if stop.type == 1: 
                delay = max(0, arrival - stop.time_window)
                total_delay += delay
        return total_delay

    def try_insert(self, v_idx, r_idx):
        """
        核心复现：Cheapest Insertion Heuristic
        尝试把当前订单(Pickup @ Restaurant, Delivery @ Customer)插入到车辆 v 的路线中
        返回：(增加的延误, 新的路线)
        """
        v = self.vehicles[v_idx]
        r = config.RESTAURANTS[r_idx]
        req = self.current_req
        
        # 构造两个新Stop
        stop_p = Stop(r['loc'], 0, self.time, req['id']) # Pickup
        stop_d = Stop(req['loc'], 1, self.time + config.PROMISED_TIME, req['id']) # Delivery
        
        best_cost = float('inf')
        best_route = None
        
        # 原始路线延误 (Baseline)
        base_delay = self.calculate_route_delay(v, v.route)
        
        # 遍历所有可能的插入位置 (i: Pickup位置, j: Delivery位置)
        # 约束: i < j (必须先取后送)
        n = len(v.route)
        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                if len(v.route) + 2 > config.MAX_STOPS: break # 简化：防止爆炸
                
                temp_route = v.route[:i] + [stop_p] + v.route[i:j-1] + [stop_d] + v.route[j-1:]
                
                new_delay = self.calculate_route_delay(v, temp_route)
                cost_increase = new_delay - base_delay
                
                if cost_increase < best_cost:
                    best_cost = cost_increase
                    best_route = temp_route
                    
        return best_cost, best_route

    def step(self, r_idx, v_idx):
        # 执行动作：应用最佳插入
        _, best_route = self.try_insert(v_idx, r_idx)
        
        if best_route is not None:
            self.vehicles[v_idx].route = best_route
            
        # 状态更新：时间流逝，车辆移动，移除已完成Stop
        self.time += 5.0 # 时间步长
        
        reward = 0
        for v in self.vehicles:
            # 模拟车辆移动逻辑 (简化版：每个时间步处理掉第一个Stop如果到达了)
            if v.route:
                travel = self.get_travel_time(v.loc, v.route[0].loc)
                if self.time >= v.next_free_time + travel:
                    finished_stop = v.route.pop(0)
                    v.loc = finished_stop.loc
                    v.next_free_time += travel
                    # 只有送达才有Reward(负延误)
                    if finished_stop.type == 1:
                        delay = max(0, v.next_free_time - finished_stop.time_window)
                        reward -= delay

        self.generate_request()
        return reward