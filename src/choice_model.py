import numpy as np
import config

def simulate_customer_choice(sorted_ranking_indices):
    """
    MNL 模型 (Multinomial Logit Model)
    输入: 
        sorted_ranking_indices: 排序后的餐厅 ID 列表，例如 [2, 0, 1] (餐厅2排第一)
    输出: 
        chosen_index: 用户在 sorted_ranking_indices 中的索引 (选择了第几名的餐厅)
        chosen_r_id: 被选中的真实餐厅 ID
    """
    utilities = []
    
    # 遍历排序后的餐厅列表
    for rank, r_idx in enumerate(sorted_ranking_indices):
        # 获取餐厅的基础效用 (Base Utility)
        r_info = config.RESTAURANTS[r_idx]
        base_u = r_info['base_u']
        
        # 计算总效用: Base + Rank Bias
        # rank 0 (第一名) 加成最大，越往后效用越低
        u = base_u + (rank * -0.5) # 这里直接写死 -0.5 或用 config.RANK_COEFF
        utilities.append(np.exp(u))
    
    # Softmax 概率计算
    total_u = sum(utilities)
    probs = np.array(utilities) / total_u
    
    # 根据概率随机选择
    # 注意：这里返回的是“用户选了第几名”，比如选了第0名（排第一那个）
    choice_idx_in_ranking = np.random.choice(len(sorted_ranking_indices), p=probs)
    
    # 找到对应的真实餐厅ID
    chosen_r_id = sorted_ranking_indices[choice_idx_in_ranking]
    
    return choice_idx_in_ranking, chosen_r_id