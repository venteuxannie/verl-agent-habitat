"""
数据集格式转换脚本

将原始 task_infos.json 转换为包含 task_description 的格式，
为每种任务类型（grounding, segment, 3d-box）生成独立的 JSON 文件。
"""

import json
import copy
from pathlib import Path
from tqdm import tqdm
import os

from agent_system.environments.env_package.habitat_sim.utils.third_party import call_generate_task_description
from agent_system.environments.env_package.habitat_sim.utils.constants import EVAL_DATA_PATH

def convert_task_infos(
    input_json_path: str,
    output_dir: str = None,
    task_types: list = None
):
    """
    转换任务信息文件，为每种任务类型生成带有 task_description 的新 JSON 文件。
    
    Args:
        input_json_path: 原始 task_infos.json 的路径
        output_dir: 输出目录，默认与输入文件同目录
        task_types: 要生成的任务类型列表，默认为 ["segment", "3d-box"]
    """
    if task_types is None:
        task_types = ["segment", "3d-box"]
    
    input_path = Path(input_json_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取原始 JSON 文件
    print(f"正在读取: {input_json_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    print(f"共加载 {len(tasks)} 个任务")
    
    # 为每种任务类型生成新的数据集
    for task_type in task_types:
        print(f"\n{'='*60}")
        print(f"正在处理任务类型: {task_type}")
        print(f"{'='*60}")
        
        new_tasks = []
        failed_tasks = []
        
        for i, task in enumerate(tqdm(tasks, desc=f"生成 {task_type} 任务描述")):
            # 深拷贝任务，避免修改原始数据
            new_task = copy.deepcopy(task)
            
            # 更新任务类型
            new_task["task_type"] = task_type
            
            # 获取 task_prompt
            task_prompt = task.get("task_prompt", "")
            
            try:
                # 调用 API 生成 task_description
                response = call_generate_task_description(task_prompt, task_type)
                task_description = response.get("task_description", task_prompt)
                new_task["task_description"] = task_description
            except Exception as e:
                # 如果 API 调用失败，使用 task_prompt 作为 fallback
                print(f"\n警告: 任务 {i} (task_id={task.get('task_id', i)}) API 调用失败: {e}")
                new_task["task_description"] = task_prompt
                failed_tasks.append(i)
            
            new_tasks.append(new_task)
        
        # 保存新的 JSON 文件
        output_filename = f"task_infos_{task_type}.json"
        output_path = output_dir / output_filename
        
        print(f"\n正在保存: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_tasks, f, indent=4, ensure_ascii=False)
        
        print(f"成功保存 {len(new_tasks)} 个任务到 {output_filename}")
        if failed_tasks:
            print(f"警告: {len(failed_tasks)} 个任务 API 调用失败，已使用 task_prompt 作为 task_description")
    
    print(f"\n{'='*60}")
    print("转换完成!")
    print(f"{'='*60}")


def main():
    """主函数"""
    # 默认输入路径
    input_json_path = os.path.join(EVAL_DATA_PATH, "replicacad_10-any-500-seen/task_infos.json")
    
    # 执行转换
    convert_task_infos(input_json_path)


if __name__ == "__main__":
    main()
