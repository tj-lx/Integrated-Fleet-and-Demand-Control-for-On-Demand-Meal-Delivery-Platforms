# Integrated Fleet and Demand Control (Simplified Reproduction)

这是一个基于 Python 的简化复现项目，旨在演示 **Hildebrandt et al. (2025)** 发表在 *Management Science* 上的论文 **"Integrated Fleet and Demand Control for On-Demand Meal Delivery Platforms"** 的核心算法思想。

由于没有数据，本项目通过一个简化的网格世界（Grid World）模拟，展示了如何利用深度强化学习（Deep Q-Learning）同时优化**前端餐厅排序（需求控制）**和**后端车辆派单（运力控制）**，从而在引导客户行为的同时降低物流延误。

## 🛠️ 项目环境依赖

请确保安装了 Python 3.8+。

安装依赖库：
```bash
pip install -r requirements.txt