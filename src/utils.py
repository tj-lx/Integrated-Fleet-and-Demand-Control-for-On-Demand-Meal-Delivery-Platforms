import torch
import numpy as np
import config

def extract_features(env, r_idx):
    """
    把当前环境状态转换为 Tensor，用于输入神经网络
    返回: 
    1. stops_tensor: [1, N_V, Max_Stops, Feat]
    2. mask_tensor: [1, N_V, Max_Stops]
    3. context: [1, 4]
    """
    batch_stops = []
    batch_mask = []
    
    req = env.current_req
    rest = config.RESTAURANTS[r_idx]
    
    # Context Features
    context = torch.tensor([[req['loc'][0], req['loc'][1], rest['loc'][0], rest['loc'][1]]], dtype=torch.float32)
    
    veh_stops_list = []
    veh_mask_list = []
    
    for v in env.vehicles:
        # 提取每辆车的路线特征
        stops_feat = []
        # 加入当前车辆位置作为“第0个Stop”
        stops_feat.append([0, v.loc[0], v.loc[1], 0, 0, 0]) 
        
        for s in v.route:
            # Feature: [Type, x, y, PromisedTime - Now, Is_Belong_To_Cur_Req, Load]
            # 这里简化特征，保证维度一致
            feat = [
                s.type, s.loc[0], s.loc[1], 
                s.time_window - env.time, 
                1.0 if s.belongs_to_req == req['id'] else 0.0,
                len(v.route)
            ]
            stops_feat.append(feat)
            
        # Padding
        pad_len = config.MAX_STOPS - len(stops_feat)
        mask = [0] * len(stops_feat) + [1] * pad_len # 0 valid, 1 padding
        stops_feat += [[0]*config.FEATURE_DIM] * pad_len
        
        veh_stops_list.append(stops_feat)
        veh_mask_list.append(mask)
        
    return (
        torch.tensor([veh_stops_list], dtype=torch.float32),
        torch.tensor([veh_mask_list], dtype=torch.float32),
        context
    )