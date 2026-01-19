import numpy as np
import torch
import torch.nn.functional as F

class ChoiceModel:
    def __init__(self, num_restaurants):
        # 模拟论文 Figure 4 中的参数
        # 排名越靠前(index小)，权重越高。模拟论文中的 Position Bias
        self.position_bias = np.linspace(2.0, -1.0, num_restaurants)
        
    def get_choice_probabilities(self, restaurants, display_order):
        """
        [cite_start]根据 L-MNL 模型计算选择概率 [cite: 354-360]
        Args:
            restaurants: 餐厅对象列表
            display_order: 餐厅ID的排列列表 (由Agent决定)
        """
        utilities = []
        ordered_restaurants = [restaurants[i] for i in display_order]
        
        for rank, rest in enumerate(ordered_restaurants):
            # Utility = Restaurant_Feature + Position_Bias
            # 论文公式(1)的简化版
            u = rest.base_score + self.position_bias[rank]
            utilities.append(u)
            
        probs = F.softmax(torch.tensor(utilities, dtype=torch.float32), dim=0)
        return probs, ordered_restaurants