import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # 设置可见的GPU设备

from agent_system.environments import make_envs
import ray

NUM = 2

class Config:
    class Env:
        def __init__(self):
            self.env_name = "habitat"  # 环境名称，例如 "gym_cards/blackjack"
            self.rollout = type("Rollout", (object,), {"n": NUM})  # rollout 参数，n 必须是整数
            self.seed = 0  # 随机种子
            self.dataset_name = "HM3D"  # 数据集名称（仅适用于某些环境）
            self.max_steps = 10  # 历史长度（适用于某些环境）

    class Data:
        def __init__(self):
            self.train_batch_size = NUM  # 训练环境的批量大小
            self.val_batch_size = NUM  # 验证环境的批量大小

    def __init__(self):
        self.env = self.Env()
        self.data = self.Data()

# 示例使用
config = Config()
envs, val_envs = make_envs(config)
# breakpoint()