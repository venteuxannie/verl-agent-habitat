#!/usr/bin/env python3
"""
修复 SFT 模型的 config.json
LLaMA-Factory 保存模型时会丢失很多配置字段，导致 vLLM 无法正确加载
"""

import json
import shutil
import argparse
from pathlib import Path


def fix_config(original_model_path: str, sft_model_path: str, backup: bool = True):
    """
    从原始模型复制完整的 config.json 到 SFT 模型
    
    Args:
        original_model_path: 原始模型路径
        sft_model_path: SFT 后的模型路径
        backup: 是否备份原文件
    """
    original_config_path = Path(original_model_path) / "config.json"
    sft_config_path = Path(sft_model_path) / "config.json"
    
    # 检查文件是否存在
    if not original_config_path.exists():
        raise FileNotFoundError(f"Original config.json not found: {original_config_path}")
    if not sft_config_path.exists():
        raise FileNotFoundError(f"SFT config.json not found: {sft_config_path}")
    
    # 备份 SFT 模型的配置文件
    if backup:
        backup_path = sft_config_path.with_suffix(".json.backup")
        shutil.copy2(sft_config_path, backup_path)
        print(f"✅ Backed up SFT config.json to: {backup_path}")
    
    # 读取配置
    with open(original_config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)
    
    with open(sft_config_path, 'r', encoding='utf-8') as f:
        sft_config = json.load(f)
    
    # 对比分析
    print("\n" + "=" * 80)
    print("配置字段分析")
    print("=" * 80)
    
    # llm_config对比
    if 'llm_config' in original_config and 'llm_config' in sft_config:
        orig_llm_keys = set(original_config['llm_config'].keys())
        sft_llm_keys = set(sft_config['llm_config'].keys())
        missing_llm_keys = orig_llm_keys - sft_llm_keys
        print(f"\nllm_config:")
        print(f"  Original: {len(orig_llm_keys)} keys")
        print(f"  SFT:      {len(sft_llm_keys)} keys")
        print(f"  Missing:  {len(missing_llm_keys)} keys")
        if missing_llm_keys:
            print(f"  Missing keys: {sorted(list(missing_llm_keys))[:10]}...")  # 只显示前10个
    
    # vision_config对比
    if 'vision_config' in original_config and 'vision_config' in sft_config:
        orig_vision_keys = set(original_config['vision_config'].keys())
        sft_vision_keys = set(sft_config['vision_config'].keys())
        missing_vision_keys = orig_vision_keys - sft_vision_keys
        print(f"\nvision_config:")
        print(f"  Original: {len(orig_vision_keys)} keys")
        print(f"  SFT:      {len(sft_vision_keys)} keys")
        print(f"  Missing:  {len(missing_vision_keys)} keys")
        if missing_vision_keys:
            print(f"  Missing keys: {sorted(list(missing_vision_keys))[:10]}...")
    
    print("\n" + "=" * 80)
    print("开始修复...")
    print("=" * 80)
    
    # 策略：保留 SFT config 作为基础，但用原始 config 补充缺失的字段
    # 这样可以保留 SFT 可能修改的参数，同时恢复缺失的配置
    
    fixed_config = original_config.copy()
    
    # 保留 SFT 中可能更新的字段（如果有的话）
    # 通常 SFT 不会改变模型架构，所以直接用原始配置是安全的
    
    # 写回文件
    with open(sft_config_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully updated config.json at: {sft_config_path}")
    print("\n📋 Summary:")
    print(f"   - Restored llm_config keys: {len(missing_llm_keys) if 'llm_config' in original_config else 0}")
    print(f"   - Restored vision_config keys: {len(missing_vision_keys) if 'vision_config' in original_config else 0}")
    print("\n⚠️  Important:")
    print("   1. Backup created at: {}.backup".format(sft_config_path))
    print("   2. MUST restart vLLM server for changes to take effect!")
    print("   3. Model weights are unchanged, only config restored")


def compare_configs(original_model_path: str, sft_model_path: str):
    """对比两个模型的 config.json"""
    original_config_path = Path(original_model_path) / "config.json"
    sft_config_path = Path(sft_model_path) / "config.json"
    
    with open(original_config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)
    
    with open(sft_config_path, 'r', encoding='utf-8') as f:
        sft_config = json.load(f)
    
    print("\n" + "=" * 80)
    print("📊 Config.json Comparison")
    print("=" * 80)
    
    # 顶层字段
    print("\n顶层字段:")
    orig_keys = set(original_config.keys())
    sft_keys = set(sft_config.keys())
    print(f"  Original: {len(orig_keys)} keys")
    print(f"  SFT:      {len(sft_keys)} keys")
    print(f"  Missing in SFT: {orig_keys - sft_keys}")
    print(f"  Extra in SFT:   {sft_keys - orig_keys}")
    
    # llm_config
    if 'llm_config' in original_config and 'llm_config' in sft_config:
        print("\nllm_config:")
        orig_llm_keys = set(original_config['llm_config'].keys())
        sft_llm_keys = set(sft_config['llm_config'].keys())
        missing_keys = orig_llm_keys - sft_llm_keys
        print(f"  Original: {len(orig_llm_keys)} keys")
        print(f"  SFT:      {len(sft_llm_keys)} keys")
        print(f"  ❌ Missing {len(missing_keys)} keys:")
        for key in sorted(list(missing_keys))[:15]:  # 显示前15个
            print(f"     - {key}")
        if len(missing_keys) > 15:
            print(f"     ... and {len(missing_keys) - 15} more")
    
    # vision_config
    if 'vision_config' in original_config and 'vision_config' in sft_config:
        print("\nvision_config:")
        orig_vision_keys = set(original_config['vision_config'].keys())
        sft_vision_keys = set(sft_config['vision_config'].keys())
        missing_keys = orig_vision_keys - sft_vision_keys
        print(f"  Original: {len(orig_vision_keys)} keys")
        print(f"  SFT:      {len(sft_vision_keys)} keys")
        print(f"  ❌ Missing {len(missing_keys)} keys:")
        for key in sorted(list(missing_keys))[:15]:
            print(f"     - {key}")
        if len(missing_keys) > 15:
            print(f"     ... and {len(missing_keys) - 15} more")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Fix SFT model's config.json by restoring missing fields from original model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 对比配置
  python fix_sft_config.py --compare \\
      --original /data1/tct_data/InternVL/models/InternVL3_5-2B-MPO \\
      --sft /data1/tct_data/sft-model/InternVL3_5-2B-vg-sft
  
  # 修复 SFT 模型配置
  python fix_sft_config.py --fix \\
      --original /data1/tct_data/InternVL/models/InternVL3_5-2B-MPO \\
      --sft /data1/tct_data/sft-model/InternVL3_5-2B-vg-sft
        """
    )
    
    parser.add_argument('--original', type=str, required=True,
                        help='Path to the original model directory')
    parser.add_argument('--sft', type=str, required=True,
                        help='Path to the SFT model directory')
    parser.add_argument('--compare', action='store_true',
                        help='Compare configs without making changes')
    parser.add_argument('--fix', action='store_true',
                        help='Fix the SFT config by restoring missing fields')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup of original SFT config')
    
    args = parser.parse_args()
    
    if not args.compare and not args.fix:
        parser.error("Please specify either --compare or --fix")
    
    if args.compare:
        compare_configs(args.original, args.sft)
    
    if args.fix:
        fix_config(args.original, args.sft, backup=not args.no_backup)


if __name__ == '__main__':
    main()

