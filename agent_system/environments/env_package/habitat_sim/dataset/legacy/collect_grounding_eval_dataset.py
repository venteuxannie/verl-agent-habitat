import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from typing import Dict, List
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import draw_bbox_with_text, get_dataset_subfolder
from agent_system.environments.env_package.habitat_sim.utils.constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE
from agent_system.environments.env_package.habitat_sim.utils.constants import SCENE_DATA_PATH, EVAL_DATA_PATH, HM3D_TRAINING_SCENES

class NumpyEncoder(json.JSONEncoder):
    """
    一个特殊的JSON Encoder，用于处理Numpy数据类型。
    当遇到不认识的类型时，会调用default方法。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # 将Numpy整数转换为Python int
        elif isinstance(obj, np.floating):
            return float(obj) # 将Numpy浮点数转换为Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # 将Numpy数组转换为Python list
        return super(NumpyEncoder, self).default(obj)
    
def get_scene_path(subfolders, dataset_name, eval_id=0):
    """Constructs the full path to a habitat scene file."""
    data_path = SCENE_DATA_PATH # <-- Note: Hardcoded path
    if dataset_name == "HM3D":
        dataset_path = os.path.join(data_path, "hm3d/train")
        scene = subfolders[eval_id % len(subfolders)]
        id = scene.split('-')[1]
        scene_path = f"{scene}/{id}.basis.glb"
        scene_id = os.path.join(dataset_path, scene_path)
    elif dataset_name == "ReplicaCAD":
        scene_name = subfolders[eval_id % len(subfolders)]
        scene_id = os.path.join(data_path, "replica_cad/configs/scenes", scene_name)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return scene_id

def reshape_bbox_xxyy(bbox, original_size=GROUNDINGDINO_ORIGINAL_SIZE, new_size=GROUNDINGDINO_NEW_SIZE):
    """
    (针对GroundingDINO的输出) 根据图像尺寸变化对bbox进行转换。

    参数:
        bbox (tuple): 原始bbox，格式为 (x1, y1, x2, y2)。
        original_size (tuple): 原始图像尺寸 (宽, 高)。
        new_size (tuple): 新图像尺寸 (宽, 高)。

    返回:
        tuple: 转换后的bbox，格式为 (x, x, y, y)。
    """
    # 计算宽度和高度的缩放比例
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    # 对bbox的坐标进行缩放
    x1, y1, x2, y2 = bbox
    x1_new = x1 * scale_x
    y1_new = y1 * scale_y
    x2_new = x2 * scale_x
    y2_new = y2 * scale_y

    return (x1_new, x2_new, y1_new, y2_new)

dataset_name = "HM3D"
dir_path = EVAL_DATA_PATH
eval_data_name = "hm3d_10-any-500-seen"
eval_data_path = os.path.join(dir_path, eval_data_name)
if not os.path.exists(eval_data_path):
    os.makedirs(eval_data_path)
image_save_path = os.path.join(eval_data_path, "images")
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

seed = 0
# scenes_size = 10
# max_scene_instance = 20
scenes_size = 10
max_scene_instance = 50
max_step_length = 10

env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
print("Habitat environment created.")

# 1. 初始化一个列表，用于存储所有info信息
all_task_infos = []
# 2. 初始化任务ID计数器
task_id_counter = 0

scene_subfolder = env.scene_subfolder # Traning Scenes
# scene_subfolder = get_dataset_subfolder(dataset_name)[10:20] # New Scenes
# --- 你原来的代码逻辑 ---
for i in tqdm(range(scenes_size), desc="Processing Scenes"):
    special_scene_id = get_scene_path(scene_subfolder, dataset_name, eval_id=i)
    for j in range(max_scene_instance):
        obs, info = env.reset(seed=seed, is_unique=True, sync_info=None, special_scene_id=special_scene_id, task_type="grounding")
        info["task_type"] = "any"
        # 3. 为当前的info字典添加一个task_id
        info['task_id'] = task_id_counter
        
        # 4. 将更新后的info添加到总列表中
        all_task_infos.append(info)
        
        # 5. 递增ID，为下一个info做准备
        task_id_counter += 1

        # 你原来的绘图逻辑可以保持不变
        annotation_obs = draw_bbox_with_text(obs, info.get("gt")["bbox_gt"], text="GT", color="red", title=info.get("task_prompt"))
        annotation_obs = draw_bbox_with_text(annotation_obs, reshape_bbox_xxyy(info.get("pred")), text="VG", color="green")
        annotation_obs.save(os.path.join(image_save_path, f"{task_id_counter}.png"))

# 6. 所有循环结束后，将整个列表写入JSON文件
output_filename =  os.path.join(eval_data_path, 'task_infos.json')
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(all_task_infos, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

print(f"所有 {task_id_counter} 条info已成功保存到 {output_filename}")