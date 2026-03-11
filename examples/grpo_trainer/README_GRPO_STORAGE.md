# GRPO训练输出存储使用说明

## 概述

修改后的 `run_habitat_debug.sh` 脚本现在支持在GRPO训练过程中自动存储模型输出，包括：

- **Rollout数据**: 训练过程中的模型生成内容
- **验证数据**: 验证阶段的模型输出

## 主要修改

### 1. 模型输出目录配置
```bash
# Model output storage configuration
OUTPUT_DIR="./grpo_habitat_outputs"
ROLLOUT_OUTPUT_DIR="$OUTPUT_DIR/rollouts"
VALIDATION_OUTPUT_DIR="$OUTPUT_DIR/validations"
```

### 2. 自动创建目录
脚本会自动创建所需的输出目录。

### 3. 训练配置参数
添加了以下配置参数：
- `trainer.rollout_data_dir`: 存储rollout数据
- `trainer.validation_data_dir`: 存储验证数据
- `trainer.rollout_log_freq=10`: 每10步记录一次rollout数据
- `trainer.log_val_generations=5`: 记录5个验证样本

## 使用方法

### 启动训练
```bash
cd /data1/tct_data/verl-agent/examples/grpo_trainer
bash run_habitat_debug.sh
```

### 调整记录频率
你可以通过修改脚本中的 `ROLLOUT_LOG_FREQ` 变量来调整rollout数据的记录频率：

```bash
# 在脚本中修改这一行
ROLLOUT_LOG_FREQ=10  # 每10步记录一次rollout数据

# 不同频率的建议：
ROLLOUT_LOG_FREQ=1   # 每步都记录（最详细，但占用空间大）
ROLLOUT_LOG_FREQ=5   # 每5步记录一次（平衡详细度和存储空间）
ROLLOUT_LOG_FREQ=10  # 每10步记录一次（默认设置）
ROLLOUT_LOG_FREQ=20  # 每20步记录一次（节省存储空间）
```

### 查看存储的数据
```bash
# 查看目录结构
ls -la ./grpo_habitat_outputs/

# 查看rollout数据
ls -la ./grpo_habitat_outputs/rollouts/

# 查看单个文件内容
head -n 1 ./grpo_habitat_outputs/rollouts/*.jsonl | jq .
```

### 分析存储的数据
```bash
# 基本分析
python analyze_grpo_outputs.py --output-dir ./grpo_habitat_outputs

# 生成训练进度图表
python analyze_grpo_outputs.py --output-dir ./grpo_habitat_outputs --plot

# 显示更多样本
python analyze_grpo_outputs.py --output-dir ./grpo_habitat_outputs --show-samples 10
```

## 存储的数据格式

### Rollout数据 (JSONL格式)
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

### 文件命名规则
- Rollout数据: `{step}.jsonl` (例如: `100.jsonl`, `200.jsonl`)
- 验证数据: `{step}.jsonl` (例如: `10.jsonl`, `20.jsonl`)

## 分析工具功能

`analyze_grpo_outputs.py` 脚本提供以下分析功能：

1. **基本统计**:
   - 平均分数和标准差
   - 优势值和回报值统计
   - 动作有效性比例
   - 数据源分布

2. **训练进度分析**:
   - 分数随训练步数的变化
   - 优势值随训练步数的变化
   - 分数和优势值的分布直方图

3. **样本展示**:
   - 显示具体的输入输出样本
   - 展示相关的分数和元数据

## 注意事项

1. **存储空间**: 存储详细输出会占用较多磁盘空间，建议定期清理旧文件
2. **性能影响**: 启用输出存储会略微影响训练速度
3. **文件格式**: 输出文件为JSONL格式，便于逐行读取和分析
4. **并发安全**: 存储过程是线程安全的，支持多进程训练

## 故障排除

### 常见问题

1. **目录权限问题**
   ```bash
   chmod 755 ./grpo_habitat_outputs/
   ```

2. **磁盘空间不足**
   ```bash
   df -h  # 检查磁盘空间
   du -sh ./grpo_habitat_outputs/  # 检查输出目录大小
   ```

3. **JSON解析错误**
   ```bash
   # 检查文件格式
   head -n 1 ./grpo_habitat_outputs/rollouts/*.jsonl | python -m json.tool
   ```

### 调试信息

训练过程中会输出以下信息：
```
Output directories created:
  Rollout outputs: ./grpo_habitat_outputs/rollouts
  Validation outputs: ./grpo_habitat_outputs/validations
  Checkpoints: ./grpo_habitat_outputs/checkpoints

GRPO Training - Batch size: 8, Token-level rewards shape: torch.Size([8, 128]), Response mask shape: torch.Size([8, 128])
GRPO Training - Unique UIDs: 8
GRPO Training - Unique trajectory UIDs: 2
Dumped generations to ./grpo_habitat_outputs/rollouts/100.jsonl
```

这些信息有助于监控GRPO训练的状态和存储过程。
