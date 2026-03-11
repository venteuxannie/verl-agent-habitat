# Habitat环境动作序列测试工具

**action_test.py** - 简洁版的Habitat环境动作序列测试工具

## 功能特色

- ✅ **简洁易读**: 只包含核心功能，无复杂日志记录
- ✅ **实时监控**: 控制台实时显示IoU和置信度变化
- ✅ **任务指定**: 使用task_id指定特定任务进行测试
- ✅ **动作序列**: 支持预设动作序列执行

## 支持的动作

| 动作名称 | 说明 |
|---------|------|
| `move_forward` | 向前移动 |
| `turn_left` | 向左转 |
| `turn_right` | 向右转 |
| `look_up` | 向上看 |
| `look_down` | 向下看 |
| `stop` | 停止 |

## 使用方法

为脚本提供必需的`model_path`参数（兼容性要求），但实际不使用模型推理：

### 基本用法

```bash
python action_test.py \
  --model_path /dummy/path \
  --action_sequence "move_forward,turn_left,turn_right" \
  --task_id 18
```

### 示例命令

1. **测试转向动作**
```bash
python action_test.py \
  --model_path /dummy/path \
  --action_sequence "turn_left,turn_right,turn_right,turn_left" \
  --task_id 0
```

2. **测试复合动作**
```bash
python action_test.py \
  --model_path /dummy/path \
  --action_sequence "move_forward,move_forward,turn_left,look_up" \
  --task_id 25
```

3. **测试垂直视角调整**
```bash
python action_test.py \
  --model_path /dummy/path \
  --action_sequence "look_up,look_down,look_down,look_up" \
  --task_id 42
```

## 输出示例

```
支持的动作: move_forward, turn_left, turn_right, look_up, look_down, stop

=== 任务 18: 找到红色的椅子 ===
场景: apartment_1/apartment_1.basis.glb
初始 IoU: 0.0000, 初始 Conf: 0.7500
动作序列: turn_left → turn_right → turn_right → turn_left

开始执行动作序列...
步骤 1: 执行动作 'turn_left'
        IoU: 0.0000 (变化: +0.0000)
        Conf: 0.7800 (变化: +0.0300)
        奖励: -0.01
步骤 2: 执行动作 'turn_right'
        IoU: 0.1250 (变化: +0.1250)
        Conf: 0.8200 (变化: +0.0700)
        奖励: 0.15
        ✓ 任务在步骤 2 完成!

=== 执行结果 ===
IoU: 0.0000 → 0.1250 (+0.1250)
Conf: 0.7500 → 0.8200 (+0.0700)
任务状态: 完成
执行动作数: 4
```

## 核心特性

- **实时反馈**: 每步都显示IoU、置信度和奖励
- **任务完成检测**: 当任务完成时立即停止
- **错误处理**: 自动处理无效动作（替换为随机动作）
- **简洁输出**: 清晰的格式化输出，易于阅读

## 参数说明

- `--action_sequence`: 必需，逗号分隔的动作序列
- `--task_id`: 可选，默认为0（任务ID范围：0-99）
- `--model_path`: 必需（兼容性），但实际不使用

## 与原始eval_notebook.py的区别

| 特性 | action_test.py | eval_notebook.py |
|------|----------------|------------------|
| 复杂度 | 简洁 | 复杂 |
| 日志文件 | 无 | HTML+TXT |
| 图像保存 | 无 | Base64编码 |
| 依赖 | 最小 | 较多 |
| 可读性 | 高 | 中等 |
| 用途 | 快速测试 | 完整评估 |










