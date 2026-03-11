# Habitat环境动作序列测试脚本使用说明

## 功能概述

修改后的`eval_notebook.py`脚本现在支持两种模式：
1. **原始模型评估模式**：使用Qwen-VL模型进行推理
2. **预设动作测试模式**：使用指定的动作序列控制agent

## 新增功能

### 预设动作测试模式

通过添加新的命令行参数，可以用预定义的动作序列控制agent运动，无需模型推理。

#### 支持的参数

- `--test_actions`: 启用预设动作测试模式
- `--task_id`: 指定要测试的特定任务ID (0-99)
- `--action_sequence`: 逗号分隔的动作列表
- `--max_steps`: 最大执行步数（可选）

#### 支持的动作

- `move_forward`: 向前移动
- `turn_left`: 向左转
- `turn_right`: 向右转
- `look_up`: 向上看
- `look_down`: 向下看
- `stop`: 停止

## 使用示例

### 1. 测试特定任务的动作序列

```bash
# 测试任务ID 18，执行左右转向序列
python eval_notebook.py \
  --model_path /path/to/qwen-vl-model \
  --test_actions \
  --task_id 18 \
  --action_sequence "turn_left,turn_right,turn_right,turn_left"

# 测试前进和转向的复合动作
python eval_notebook.py \
  --model_path /path/to/qwen-vl-model \
  --test_actions \
  --task_id 0 \
  --action_sequence "move_forward,move_forward,turn_left,look_up"

# 测试垂直视角调整
python eval_notebook.py \
  --model_path /path/to/qwen-vl-model \
  --test_actions \
  --task_id 25 \
  --action_sequence "look_up,look_down,look_down,look_up"
```

### 2. 长序列测试

```bash
# 测试更长的动作序列
python eval_notebook.py \
  --model_path /path/to/qwen-vl-model \
  --test_actions \
  --task_id 42 \
  --action_sequence "move_forward,turn_left,move_forward,turn_right,move_forward,turn_right,look_up,look_down"
```

### 3. 原始模型评估（保持不变）

```bash
# 使用模型进行推理评估
python eval_notebook.py \
  --model_path /path/to/qwen-vl-model \
  --log_filename evaluation_logs/model_eval_log.html
```

## 输出说明

### 日志文件

脚本会生成两种格式的日志：
1. **HTML格式**: 包含可视化图像和结构化信息
2. **TXT格式**: 纯文本格式，便于程序处理

### 性能指标

- **IoU变化**: 初始IoU vs 最终IoU的变化
- **Conf变化**: 置信度分数的变化
- **任务完成状态**: 是否在动作序列执行期间完成任务
- **执行步数**: 实际执行的动作数量

### 可视化内容

- 初始状态图像（带GT和VG bbox标注）
- 每个动作步骤后的环境观察
- IoU和置信度分数的实时变化
- 颜色编码的性能变化（绿色=改善，红色=下降）

## 应用场景

1. **动作序列验证**: 测试特定动作序列在不同任务上的效果
2. **环境交互测试**: 验证agent与环境的基本交互能力
3. **基准测试**: 建立动作序列性能的基准线
4. **调试工具**: 用于环境调试和问题排查

## 注意事项

1. 确保提供有效的`model_path`参数（即使不使用模型推理）
2. `task_id`范围应在0到任务总数-1之间
3. 动作名称必须精确匹配支持的动作列表
4. 日志文件会自动创建目录结构
5. 如果任务在执行过程中完成，会提前结束动作序列执行




