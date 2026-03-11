#!/usr/bin/env python3
"""
修复 SFT 模型的 tokenizer_config.json，添加缺失的 chat_template。

问题：LLaMA-Factory 进行 SFT 后，生成的 tokenizer_config.json 可能缺少 chat_template，
导致 vLLM 推理异常（重复输出、格式错误）。

解决方案：从原始模型复制 chat_template 到 SFT 模型。
"""

import json
import shutil
import argparse
from pathlib import Path


def fix_tokenizer_config(original_model_path: str, sft_model_path: str, backup: bool = True):
    """
    修复 SFT 模型的 tokenizer_config.json
    
    Args:
        original_model_path: 原始模型路径
        sft_model_path: SFT 后的模型路径
        backup: 是否备份原文件
    """
    original_config_path = Path(original_model_path) / "tokenizer_config.json"
    sft_config_path = Path(sft_model_path) / "tokenizer_config.json"
    
    # 检查文件是否存在
    if not original_config_path.exists():
        raise FileNotFoundError(f"Original tokenizer_config.json not found: {original_config_path}")
    if not sft_config_path.exists():
        raise FileNotFoundError(f"SFT tokenizer_config.json not found: {sft_config_path}")
    
    # 备份 SFT 模型的配置文件
    if backup:
        backup_path = sft_config_path.with_suffix(".json.backup")
        shutil.copy2(sft_config_path, backup_path)
        print(f"✅ Backed up SFT tokenizer_config.json to: {backup_path}")
    
    # 读取原始模型的配置
    with open(original_config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)
    
    # 读取 SFT 模型的配置
    with open(sft_config_path, 'r', encoding='utf-8') as f:
        sft_config = json.load(f)
    
    # 检查是否有 chat_template
    if 'chat_template' not in original_config:
        print("⚠️  Warning: Original model doesn't have chat_template either!")
        return
    
    if 'chat_template' in sft_config:
        print("ℹ️  SFT model already has chat_template. Checking if it's the same...")
        if sft_config['chat_template'] == original_config['chat_template']:
            print("✅ Chat templates are identical. No action needed.")
            return
        else:
            print("⚠️  Chat templates differ. Updating with original template...")
    else:
        print("🔧 SFT model is missing chat_template. Adding it now...")
    
    # 复制 chat_template
    sft_config['chat_template'] = original_config['chat_template']
    
    # 写回文件
    with open(sft_config_path, 'w', encoding='utf-8') as f:
        json.dump(sft_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully updated tokenizer_config.json at: {sft_config_path}")
    print("\n📋 Summary:")
    print(f"   - Original model: {original_model_path}")
    print(f"   - SFT model: {sft_model_path}")
    print(f"   - Chat template: {'Updated' if 'chat_template' not in sft_config else 'Replaced'}")
    print("\n⚠️  Important: Restart your vLLM server for changes to take effect!")


def compare_configs(original_model_path: str, sft_model_path: str):
    """对比两个模型的 tokenizer_config.json 的关键差异"""
    original_config_path = Path(original_model_path) / "tokenizer_config.json"
    sft_config_path = Path(sft_model_path) / "tokenizer_config.json"
    
    with open(original_config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)
    
    with open(sft_config_path, 'r', encoding='utf-8') as f:
        sft_config = json.load(f)
    
    print("\n" + "=" * 80)
    print("📊 Tokenizer Config Comparison")
    print("=" * 80)
    
    # 关键字段
    key_fields = ['chat_template', 'eos_token', 'pad_token', 'bos_token', 
                  'model_max_length', 'tokenizer_class']
    
    for field in key_fields:
        orig_val = original_config.get(field, "NOT FOUND")
        sft_val = sft_config.get(field, "NOT FOUND")
        
        if field == 'chat_template':
            # chat_template太长，只显示是否存在
            orig_display = f"EXISTS ({len(str(orig_val))} chars)" if orig_val != "NOT FOUND" else "NOT FOUND"
            sft_display = f"EXISTS ({len(str(sft_val))} chars)" if sft_val != "NOT FOUND" else "NOT FOUND"
        else:
            orig_display = str(orig_val)
            sft_display = str(sft_val)
        
        status = "✅" if orig_val == sft_val else "❌"
        print(f"\n{status} {field}:")
        print(f"   Original: {orig_display}")
        print(f"   SFT:      {sft_display}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Fix SFT model's tokenizer_config.json by copying chat_template from original model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 对比两个模型的配置
  python fix_sft_tokenizer.py --compare \\
      --original /data1/tct_data/InternVL/models/InternVL3_5-2B-MPO \\
      --sft /data1/tct_data/sft-model/InternVL3_5-2B-vg-sft
  
  # 修复 SFT 模型
  python fix_sft_tokenizer.py --fix \\
      --original /data1/tct_data/InternVL/models/InternVL3_5-2B-MPO \\
      --sft /data1/tct_data/sft-model/InternVL3_5-2B-vg-sft
        """
    )
    
    parser.add_argument('--original', type=str, required=True,
                        help='Path to the original model directory')
    parser.add_argument('--sft', type=str, required=True,
                        help='Path to the SFT model directory')
    parser.add_argument('--compare', action='store_true',
                        help='Compare tokenizer configs without making changes')
    parser.add_argument('--fix', action='store_true',
                        help='Fix the SFT tokenizer config by copying chat_template')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup of original SFT config')
    
    args = parser.parse_args()
    
    if not args.compare and not args.fix:
        parser.error("Please specify either --compare or --fix")
    
    if args.compare:
        compare_configs(args.original, args.sft)
    
    if args.fix:
        fix_tokenizer_config(args.original, args.sft, backup=not args.no_backup)


if __name__ == '__main__':
    main()

