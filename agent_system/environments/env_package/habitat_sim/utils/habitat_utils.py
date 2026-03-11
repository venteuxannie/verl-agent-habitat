import os
os.environ["HABITAT_SIM_LOG"] = "quiet"
import numpy as np
import cv2
import json
import magnum as mn
from typing import List, Union

import habitat_sim
from habitat_sim.utils import viz_utils as vut
from habitat_sim.agent import ActionSpec, ActuationSpec
from PIL import Image, ImageDraw, ImageFont

import numpy as np

import math
import random
import torch

from .constants import SENSOR_RESOLUTION, SCENE_DATA_PATH

data_path = SCENE_DATA_PATH
ReplicaCAD_object_id_to_category = None


def load_hm3d_valid_objects(scene_folder: str) -> dict:
    """
    Load preprocessed metadata for a HM3D scene.
    
    Args:
        scene_folder: Path to HM3D scene folder
        
    Returns:
        Dict with:
            - valid_semantic_ids: List of semantic IDs with preprocessed meshes
            - valid_objects: List of object info dicts
            - category_to_ids: Dict mapping category -> list of semantic IDs
    """
    import json
    metadata_path = os.path.join(scene_folder, "object_mesh", "metadata.json")
    
    result = {
        "valid_semantic_ids": [],
        "valid_objects": [],
        "category_to_ids": {}
    }
    
    if not os.path.exists(metadata_path):
        print(f"Warning: Preprocessed metadata not found for {scene_folder}")
        return result
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    valid_objects = metadata.get("valid_objects", [])
    result["valid_objects"] = valid_objects
    result["valid_semantic_ids"] = [obj["semantic_id"] for obj in valid_objects]
    
    # Build category to IDs mapping
    for obj in valid_objects:
        cat = obj["category"].lower()
        if cat not in result["category_to_ids"]:
            result["category_to_ids"][cat] = []
        result["category_to_ids"][cat].append(obj["semantic_id"])
    
    return result


def get_hm3d_scene_folder(scene_id: str) -> str:
    """
    Extract the HM3D scene folder path from a scene_id.
    
    Args:
        scene_id: Full path to the .basis.glb file
                  e.g., f"{SCENE_DATA_PATH}/hm3d/train/00009-vLpv2VX547B/vLpv2VX547B.basis.glb"
    
    Returns:
        Path to the scene folder
        e.g., f"{SCENE_DATA_PATH}/hm3d/train/00009-vLpv2VX547B"
    """
    # scene_id is the full path to .basis.glb
    # We need the parent folder
    return os.path.dirname(scene_id)

def create_hm3d_simulator(dataset_name: str, scene_id: str, enable_semantic: bool = True, enable_depth: bool = True) -> habitat_sim.Simulator:
    # 配置HM3D仿真器
    sim_cfg = habitat_sim.SimulatorConfiguration()
    if dataset_name == "HM3D":
        sim_cfg.scene_dataset_config_file = os.path.join(
            data_path, "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        )
    elif dataset_name == "ReplicaCAD":
        sim_cfg.scene_dataset_config_file = os.path.join(
            data_path, "replica_cad/replicaCAD.scene_dataset_config.json"
        )
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = 0

    # 传感器配置
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sensors = []
    # Settings
    # resolution = [640, 800]
    resolution = SENSOR_RESOLUTION
    sensor_height = 1
    # RGB传感器
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = resolution
    rgb_spec.position = [0.0, sensor_height, 0.0]
    sensors.append(rgb_spec)
    # 语义传感器
    if enable_semantic:
        semantic_spec = habitat_sim.CameraSensorSpec()
        semantic_spec.uuid = "semantic"
        semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_spec.resolution = resolution
        semantic_spec.position = [0.0, sensor_height, 0.0]
        sensors.append(semantic_spec)
    # 深度传感器
    if enable_depth:
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = resolution
        depth_spec.position = [0.0, sensor_height, 0.0]
        sensors.append(depth_spec)
    
    agent_cfg.sensor_specifications = sensors

    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=10.0)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=10.0)),
        "look_up": ActionSpec("look_up", ActuationSpec(amount=10.0)),
        "look_down": ActionSpec("look_down", ActuationSpec(amount=10.0)),
    }
    # 创建仿真器实例
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

def choose_an_instance(obs, target_ids, min_area, max_area, object_id_to_category):
    semantic_np = np.array(obs["semantic"])

    id_list = []
    for obj_id in target_ids:
        if obj_id not in object_id_to_category:
            continue
            
        # 生成mask
        mask = (semantic_np == obj_id)
        # 计算bbox面积
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            pixel_area = int((xmax - xmin + 1) * (ymax - ymin + 1))  # 计算bbox面积
        else:
            pixel_area = 0
        # category = object_id_to_category[obj_id]
        
        if min_area <= pixel_area <= max_area:
            id_list.append(obj_id)

    if id_list:
        instance_id = random.choice(list(id_list))
    else:
        instance_id = None

    return instance_id

def get_dataset_subfolder(dataset_name):
    if dataset_name == "HM3D":
        json_path = os.path.join(data_path, "hm3d/hm3d_annotated_basis.scene_dataset_config.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        subfolders = []
        for line in data["stages"]["paths"][".glb"]:
            item = line.split('/')
            if item[0] == "train" or item[0] == "val":
                subfolders.append(item[0] + '/' + item[1])
    elif dataset_name == "ReplicaCAD":
        folder_path = os.path.join(data_path, "replica_cad/configs/scenes")
        subfolders = [file for file in os.listdir(folder_path) if file.endswith('.json') and os.path.isfile(os.path.join(folder_path, file))]
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    # random.shuffle(subfolders)  # 随机打乱 subfolders
    return subfolders

def get_random_scene_path(subfolders, dataset_name):
    if dataset_name == "HM3D":
        dataset_path = os.path.join(data_path, "hm3d/train")
        scene = random.choice(subfolders)
        id = scene.split('-')[1]
        scene_path = f"{scene}/{id}.basis.glb"
        scene_id = os.path.join(dataset_path, scene_path)   # glb文件路径 
    elif dataset_name == "ReplicaCAD":
        scene_name = random.choice(subfolders)
        scene_id = os.path.join(data_path, "replica_cad/configs/scenes", scene_name) # json文件路径
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return scene_id

def get_gt_bbox(obs, instance_id, category=None):
    '''
    Output: obs_pil, bbox_gt, pixel_area
    '''
    # obs 预处理
    rgb_pil = vut.observation_to_image(obs["rgb"], "color").convert("RGB")
    semantic_np = np.array(obs["semantic"])

    # get obj_mask
    mask = (semantic_np == instance_id)

    # 计算边界框:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        pixel_area = int((xmax - xmin + 1) * (ymax - ymin + 1))  # 计算bbox面积
    else:
        return rgb_pil, [0, 0, 0, 0], 0

    return rgb_pil, (xmin, xmax, ymin, ymax), pixel_area

def draw_bbox_with_text(image: Image.Image, bbox_gt: tuple, text: str = "gt", color: str = "red", title: str = None, width: int = 3) -> Image.Image:
    """
    在给定的 PIL 图像上绘制 bbox_gt，并附加文本。

    参数:
        image (Image.Image): 输入的 PIL 图像。
        bbox_gt (tuple): 包含边界框的坐标 (xmin, xmax, ymin, ymax)。
        text (str): 要附加的文本，默认为 "gt"。
        color (str): 边界框的颜色，默认为红色。
        width (int): 边界框的线宽，默认为 3。

    返回:
        Image.Image: 绘制了边界框和文本的 PIL 图像。
    """
    # Resize 图像到 800×640
    # image = image.resize((800, 640))
    image = image.resize(SENSOR_RESOLUTION)

    # 创建一个可编辑的图像副本
    draw = ImageDraw.Draw(image)

    # 提取 bbox_gt 的坐标
    xmin, xmax, ymin, ymax = bbox_gt

    # 绘制边界框
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=width)

    # 绘制文本
    font = ImageFont.load_default()  # 使用默认字体
    text_position = (xmin + 5, ymin + 5)  # 文本位置
    draw.text(text_position, text, fill=color, font=font)
    
    # 绘制标题在左上角
    if title != None:
        title_position = (10, 10)  # 标题位置
        draw.text(title_position, title, fill="red", font=font)

    return image

def get_an_instance(
    dataset_name: str,
    sim: habitat_sim.Simulator,
    target_categories: Union[str, List[str]] = ["sofa", "chair", "table"],
    valid_semantic_ids: List[int] = None
):
    '''
    Output: hm3d_category, instance_id, obs_rgb, bbox_gt, info
    
    Args:
        dataset_name: "HM3D" or "ReplicaCAD"
        sim: habitat_sim.Simulator instance
        target_categories: List of target category names
        valid_semantic_ids: (HM3D only) List of valid semantic IDs with preprocessed meshes.
                           If provided, only these objects will be considered.
    
    Note: 此函数会一直尝试直到找到符合条件的目标实例才返回
    '''
    global ReplicaCAD_object_id_to_category  # 声明为全局变量
    
    target_category, hm3d_category, instance_id, info = None, None, None, None
    # 统一输入格式为列表
    if isinstance(target_categories, str):
        target_categories = [target_categories]
    
    # 转换为小写便于匹配
    target_categories = [cat.lower() for cat in target_categories]
    
    # 获取语义场景
    if dataset_name == "HM3D":
        semantic_scene = sim.semantic_scene
        
        # 构建语义映射字典
        object_id_to_category = {
            obj.semantic_id: obj.category.name().lower()
            for obj in semantic_scene.objects
            if obj.category is not None
        }
        
        # Filter to only valid (preprocessed) semantic IDs if provided
        if valid_semantic_ids is not None:
            valid_set = set(valid_semantic_ids)
            object_id_to_category = {
                k: v for k, v in object_id_to_category.items()
                if k in valid_set
            }
            if len(object_id_to_category) == 0:
                print(f"Warning: No valid preprocessed objects found for target categories")
    elif dataset_name == "ReplicaCAD":
        if ReplicaCAD_object_id_to_category is None:
            # 加载ReplicaCAD的object_id到category的映射
            with open(os.path.join(data_path, "replica_cad/configs/ssd/replicaCAD_semantic_lexicon.json"), 'r') as f:
                data = json.load(f)
            ReplicaCAD_object_id_to_category = {
                item["id"]: item["name"].lower()
                for item in data["classes"]
            }
        object_id_to_category = ReplicaCAD_object_id_to_category
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    while True:
        # 随机初始化Agent位置
        navigable_point = sim.pathfinder.get_random_navigable_point()
        random_rotation = np.random.uniform(0, 2 * math.pi)
        
        # 设置Agent状态
        agent = sim.get_agent(0)
        agent_state = agent.get_state()
        agent_state.position = navigable_point
        rotation_quat = mn.Quaternion.rotation(
            mn.Rad(random_rotation), mn.Vector3(0, 1, 0)
        )
        agent_state.rotation = [
            rotation_quat.vector.x,
            rotation_quat.vector.y,
            rotation_quat.vector.z,
            rotation_quat.scalar
        ]
        agent.set_state(agent_state)
        
        # 获取观测数据
        obs = sim.get_sensor_observations()

        rgb = vut.observation_to_image(obs["rgb"], "color")
        semantic = vut.observation_to_image(obs["semantic"], "semantic")

        # HM3D场景空洞检测：检测纯黑区域（RGB==0）比例
        if dataset_name == "HM3D":
            rgb_np = np.array(obs["rgb"])[:, :, :3]  # 取RGB通道，忽略alpha
            black_mask = (rgb_np[:, :, 0] == 0) & (rgb_np[:, :, 1] == 0) & (rgb_np[:, :, 2] == 0)
            black_ratio = np.sum(black_mask) / black_mask.size
            if black_ratio > 0.02:  # 阈值2%
                continue

        # 分析语义信息
        semantic = obs["semantic"]
        unique_ids = np.unique(semantic)
        
        # 获取当前视角可见物体类别
        current_categories = set()
        for obj_id in unique_ids:
            if obj_id in object_id_to_category:
                current_categories.add(object_id_to_category[obj_id])
        
        # 检测目标类别
        matched_categories = current_categories.intersection(target_categories)
        
        if matched_categories:
            random_count = 0
            while (instance_id == None) and (random_count < 5):
                # Choose one category
                target_category = random.choice(list(matched_categories))

                # Get all semantic ids of this category
                target_ids = [
                    obj_id for obj_id, category in object_id_to_category.items()
                    if category == target_category
                ]

                # Choose one instance
                width, height = rgb.size
                min_area = 0.01*width*height
                max_area = 0.04*width*height
                instance_id = choose_an_instance(obs, target_ids, min_area, max_area, object_id_to_category)

                random_count += 1
            # Attach annotation
            if instance_id != None:
                if dataset_name == "HM3D" or dataset_name == "ReplicaCAD":
                    obs_rgb, bbox_gt, pix_area = get_gt_bbox(obs, instance_id)
                    info = {
                        "coords": list(navigable_point),
                        "rotation": float(random_rotation),
                    }
                    break
                else:
                    raise ValueError(f"Unsupported dataset name: {dataset_name}")

    #TODO: 这部分后续得拆出去或者通用化，之后会更换检测器
    # 获取YOLO对应的id，方便后续计算reward
    if target_category != None:
        hm3d_category = target_category

    return hm3d_category, instance_id, obs_rgb, bbox_gt, info

def sync_agent_state(sim, agent_pos, render=True):
    agent = sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = np.array(agent_pos["coords"], dtype=np.float32)
    rotation_quat = mn.Quaternion.rotation(
        mn.Rad(agent_pos["rotation"]), mn.Vector3(0, 1, 0)
    )
    agent_state.rotation = [
        rotation_quat.vector.x,
        rotation_quat.vector.y,
        rotation_quat.vector.z,
        rotation_quat.scalar
    ]
    agent.set_state(agent_state)
    
    if render:
        # 获取观测数据
        obs = sim.get_sensor_observations()
        obs_rgb = vut.observation_to_image(obs["rgb"], "color").convert("RGB")
        return obs_rgb


def draw_reward_on_PIL(action, reward, conf_score, target_category, img):
    # 2. 创建一个绘图对象
    draw = ImageDraw.Draw(img)

    # 3. 选择字体 (确保你的系统有这个字体，或者提供字体文件路径)
    font = ImageFont.load_default()  # Windows 通常有 Arial 字体
    # 如果报错，可以尝试 `ImageFont.load_default()` 使用默认字体

    # 4. 在图像上绘制文本
    text = f"Action: {action},  Reward: {round(reward, 5)}({round(conf_score, 5)}),  Category: {target_category}"
    position = (20, 20)  # 文本起始位置 (x, y)
    text_color = "red"  # 文字颜色

    draw.text(position, text, fill=text_color, font=font)

    return img

def preprocess_obs(obs: Image) -> torch.tensor:
    image_np = np.array(obs)
    tensor_image = torch.from_numpy(image_np)

    # # 如果是 RGB 图像，需要确保顺序是 (C, H, W)
    # # 对于 RGB 图像，PIL 图像的 shape 是 (H, W, C)，需要转置
    # tensor_image = tensor_image.permute(2, 0, 1)  # 转换为 (C, H, W)

    # 使用 unsqueeze() 方法在第 0 维（最前面）增加一个维度
    tensor_with_batch = tensor_image.unsqueeze(0)

    return tensor_with_batch

def reverse_preprocess(tensor_with_batch: torch.Tensor) -> Image:
    # 移除 batch 维度
    tensor_image = tensor_with_batch.squeeze(0)
    # 将 tensor 转换为 numpy 数组，如果 tensor 在 GPU 上需要先转到 CPU
    image_np = tensor_image.cpu().numpy()
    # 如果需要，确保数据类型为 uint8
    image_np = image_np.astype(np.uint8)
    # 使用 PIL 将 numpy 数组转换为图像
    obs = Image.fromarray(image_np)
    return obs