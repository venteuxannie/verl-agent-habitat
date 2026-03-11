# GRPO训练模型输出存储功能

本功能为GRPO训练添加了增强的模型输出存储能力，可以保存训练过程中的详细信息用于分析和调试。

## 功能特性

### 1. 基础存储功能
- 存储输入文本（prompts）
- 存储输出文本（responses）
- 存储奖励分数（scores）
- 存储训练步数（step）

### 2. GRPO增强存储功能
- **优势值（Advantages）**: 存储GRPO计算的优势值
- **回报值（Returns）**: 存储GRPO计算的回报值
- **Token级奖励（Token-level rewards）**: 存储每个token的奖励
- **轨迹信息（Trajectory info）**: 存储UID和轨迹UID
- **动作有效性（Action validity）**: 存储动作是否有效
- **数据源信息（Data source）**: 存储数据来源信息

### 3. 详细日志记录
- GRPO训练过程中的批次信息
- 唯一标识符统计
- 张量形状信息

## 使用方法

### 方法1: 使用配置文件

1. 使用提供的GRPO配置文件：
```bash
python verl/trainer/main_ppo.py --config-path verl/trainer/config --config-name grpo_trainer
```

2. 或者修改现有配置文件，设置：
```yaml
trainer:
  rollout_data_dir: ./grpo_rollout_outputs  # 启用rollout输出存储
  validation_data_dir: ./grpo_validation_outputs  # 启用验证输出存储
```

### 方法2: 使用便捷脚本

```bash
# 基本使用
python run_grpo_with_output_storage.py

# 自定义配置
python run_grpo_with_output_storage.py \
    --config verl/trainer/config/grpo_trainer.yaml \
    --output-dir ./my_grpo_outputs \
    --checkpoint-dir ./my_grpo_checkpoints

# 从检查点恢复训练
python run_grpo_with_output_storage.py --resume
```

## 输出文件格式

### 存储位置
- Rollout输出: `{rollout_data_dir}/{global_step}.jsonl`
- 验证输出: `{validation_data_dir}/{global_step}.jsonl`

### JSONL文件格式
每行包含一个样本的完整信息：

```json
{
  "input": "用户输入的问题或提示",
  "output": "模型生成的回答",
  "score": 0.85,
  "step": 100,
  "advantages": 0.12,
  "returns": 0.97,
  "token_level_rewards": 0.85,
  "uid": "unique-sample-id",
  "traj_uid": "unique-trajectory-id",
  "is_action_valid": true,
  "data_source": "training_data"
}
```

## 配置参数说明

### 关键配置项

```yaml
# GRPO算法配置
algorithm:
  adv_estimator: grpo  # 使用GRPO优势估计器
  norm_adv_by_std_in_grpo: true  # GRPO中按标准差标准化优势

# 训练器配置
trainer:
  rollout_data_dir: ./grpo_rollout_outputs  # rollout输出目录
  validation_data_dir: ./grpo_validation_outputs  # 验证输出目录
  test_freq: 50  # 验证频率（每50步验证一次）
  save_freq: 100  # 保存频率（每100步保存一次）

# Actor配置（GRPO特定）
actor_rollout_ref:
  actor:
    use_kl_loss: true  # 启用KL损失
    kl_loss_coef: 0.001  # KL损失系数
    kl_loss_type: low_var_kl  # KL损失类型
  rollout:
    n: 4  # GRPO的响应数量（用于组内优势估计）
```

## 分析存储的数据

### Python分析示例

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取存储的数据
def analyze_grpo_outputs(output_file):
    data = []
    with open(output_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # 分析优势值分布
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(df['advantages'], bins=50, alpha=0.7)
    plt.title('Advantages Distribution')
    plt.xlabel('Advantage Value')
    
    # 分析奖励分数
    plt.subplot(1, 3, 2)
    plt.hist(df['score'], bins=50, alpha=0.7)
    plt.title('Reward Scores Distribution')
    plt.xlabel('Score')
    
    # 分析训练进度
    plt.subplot(1, 3, 3)
    plt.plot(df['step'], df['score'], 'o-', alpha=0.7)
    plt.title('Score vs Training Step')
    plt.xlabel('Training Step')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.show()
    
    return df

# 使用示例
df = analyze_grpo_outputs('./grpo_rollout_outputs/100.jsonl')
print(f"平均优势值: {df['advantages'].mean():.4f}")
print(f"平均奖励分数: {df['score'].mean():.4f}")
```

## 注意事项

1. **存储空间**: 存储详细输出会占用较多磁盘空间，建议定期清理旧文件
2. **性能影响**: 启用输出存储会略微影响训练速度
3. **文件格式**: 输出文件为JSONL格式，便于逐行读取和分析
4. **并发安全**: 存储过程是线程安全的，支持多进程训练

## 故障排除

### 常见问题

1. **存储目录不存在**
   - 确保指定的输出目录存在且有写权限
   - 脚本会自动创建目录

2. **配置文件错误**
   - 检查YAML文件语法
   - 确保所有必需的配置项都已设置

3. **内存不足**
   - 减少批次大小
   - 启用梯度检查点
   - 使用更少的并行进程

### 调试信息

训练过程中会输出以下调试信息：
```
GRPO Training - Batch size: 256, Token-level rewards shape: torch.Size([256, 128]), Response mask shape: torch.Size([256, 128])
GRPO Training - Unique UIDs: 64
GRPO Training - Unique trajectory UIDs: 16
Dumped generations to ./grpo_rollout_outputs/100.jsonl
```

这些信息有助于监控GRPO训练的状态和存储过程。
