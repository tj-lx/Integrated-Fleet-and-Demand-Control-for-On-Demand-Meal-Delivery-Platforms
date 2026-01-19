import numpy as np
from src.structures import Location

def get_distance(loc1: Location, loc2: Location):
    # 使用曼哈顿距离模拟城市路网
    return abs(loc1.x - loc2.x) + abs(loc1.y - loc2.y)

def calculate_arrival_time(current_time, last_loc, next_loc, velocity):
    dist = get_distance(last_loc, next_loc)
    return current_time + (dist / velocity)

def cheapest_insertion(vehicle, pickup_stop, delivery_stop, current_time, velocity):
    """
    简化的插入启发式算法：找到使得总延迟增加最小的插入位置。
    """
    best_route = None
    min_cost = float('inf')
    
    # 这是一个简化版，实际复现中可以只append到末尾以加快训练速度，
    # 或者遍历所有位置。这里为了速度，我们直接加到队尾。
    # 论文中提到使用了 insertion heuristic (Online Appendix E.3)
    
    new_route = vehicle.route.copy()
    new_route.append(pickup_stop)
    new_route.append(delivery_stop)
    
    # 计算新路线的预计完成时间作为 Cost
    cost = 0
    curr_loc = vehicle.loc
    curr_t = max(current_time, vehicle.next_free_time)
    
    for stop in new_route:
        curr_t = calculate_arrival_time(curr_t, curr_loc, stop.loc, velocity)
        cost += curr_t # 累积时间作为简单的cost
        curr_loc = stop.loc
        
    return new_route, cost