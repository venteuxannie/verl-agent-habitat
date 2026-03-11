import os
import random
from functools import partial
from typing import Dict, List

import requests
import torch
from PIL import Image
from torchvision import io
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration)

# For visualization in the notebook
from IPython.display import display, clear_output

# --- Custom Module Imports ---
# Note: Ensure these modules are in your Python path or the same directory.
import sys
sys.path.append("/data/tct/verl-agent/agent_system/environments/env_package/habitat_sim")
from utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.prompts import HABITAT_VISUAL_GROUNDING_COT_TEMPLATE
from agent_system.multi_turn_rollout.utils import process_image

from tqdm import tqdm
import json
from habitat_sim.utils import viz_utils as vut
from .third_party import call_grounding_from_pil

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_scene_path(subfolders, dataset_name, eval_id=0):
    """Constructs the full path to a habitat scene file."""
    data_path = "/data/tct/habitat/data" # <-- Note: Hardcoded path
    if dataset_name == "HM3D":
        dataset_path = os.path.join(data_path, "hm3d")
        scene = subfolders[eval_id % len(subfolders)]
        id = scene.split('/')[1].split('-')[1]
        scene_path = f"{scene}/{id}.basis.glb"
        scene_id = os.path.join(dataset_path, scene_path)
    elif dataset_name == "ReplicaCAD":
        scene_name = subfolders[eval_id % len(subfolders)]
        scene_id = os.path.join(data_path, "replica_cad/configs/scenes", scene_name)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return scene_id

def build_text_obs(infos: List[Dict]) -> List[str]:
    """Builds the text observation (prompt) for the agent."""
    postprocess_text_obs = []
    for i in range(len(infos)):
        prompt = HABITAT_VISUAL_GROUNDING_COT_TEMPLATE.format(
            task_caption=infos[i]['task_prompt'],
            conf_score=infos[i]['conf_score']
        )
        postprocess_text_obs.append(prompt)
    return postprocess_text_obs

def json_entry(id, prompt, thoughts, action):
    # 依次生成/输入id
    id_str = format(id, "06d")
    # 根据 id 自动生成图片名（假设图片名为 {id}.jpg）
    image_name = f"{id_str}.jpg"
    
    conversations = []
    value_qus = "<image>\n" + prompt
    conversations.append({
        "from": "human",
        "value": value_qus
    })
    value_ans = "{\n" + "\"thoughts\": \"" + thoughts + "\" \n}, \n"
    value_ans += "{\n" + "\"action\": \"" + action + "\" \n}"
    conversations.append({
        "from": "gpt",
        "value": value_ans
    })
    
    # 构造单条记录
    entry = {
        "image": image_name,
        "conversations": conversations
    }
    return entry, image_name

def get_object_location_description(
    image_width: int,
    image_height: int,
    bbox: List[int]
) -> str:
    """
    Generates a sentence describing an object's approximate location within an
    image, based on its bounding box.
    """
    if not all(isinstance(val, (int, float)) for val in bbox) or len(bbox) != 4:
        raise ValueError("bbox must be a list or tuple of four numbers [x1, y1, x2, y2]")

    x1, y1, x2, y2 = bbox

    # 1. Calculate the center of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 2. Determine the vertical position (top/middle/bottom)
    if center_y < image_height / 3:
        vertical_pos = "top"
    elif center_y <= image_height * 2 / 3:
        vertical_pos = "middle"
    else:
        vertical_pos = "bottom"

    # 3. Determine the horizontal position (left/center/right)
    if center_x < image_width / 3:
        horizontal_pos = "left"
    elif center_x <= image_width * 2 / 3:
        horizontal_pos = "center"
    else:
        horizontal_pos = "right"

    # 4. Map the position to a full sentence
    location_map = {
        ('top',    'left'):   "The object is in the top-left corner of the image.",
        ('top',    'center'): "The object is at the top-center of the image.",
        ('top',    'right'):  "The object is in the top-right corner of the image.",
        ('middle', 'left'):   "The object is on the left side of the image.",
        ('middle', 'center'): "The object is in the center of the image.",
        ('middle', 'right'):  "The object is on the right side of the image.",
        ('bottom', 'left'):   "The object is in the bottom-left corner of the image.",
        ('bottom', 'center'): "The object is at the bottom-center of the image.",
        ('bottom', 'right'):  "The object is in the bottom-right corner of the image.",
    }

    return location_map.get((vertical_pos, horizontal_pos), "The location of the object could not be determined.")

def get_CoT_label(
    image_width: int,
    image_height: int,
    bbox: List[int],
    conf_score: float,
    action: str
) -> str:
    """
    Generates a Chain-of-Thought (CoT) label based on the bounding box and confidence score.
    Returns:
        str: A CoT label describing the object's location and confidence score.
    """
    location_description = get_object_location_description(image_width, image_height, bbox)

    CoT = location_description
    CoT += f" The observation score is {conf_score:.3f}. "
    CoT += f"To better observe the target object, I should choose action \"{action}\"."
    return CoT

# --- Configuration ---
# dataset_name = "HM3D"
# output_name = "sft_data_vg_hm3d_CoT_2_test"
dataset_name = "ReplicaCAD"
output_name = "sft_data_vg_replica_CoT_3_test"
seed = 0
scenes_size = 2
max_scene_instance = 3
max_step_length = 10

IMAGE_FOLDER = f"/data/tct/habitat/sft_data/{output_name}"
FILE_NAME = f"/data/tct/habitat/sft_data/{output_name}.json"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# --- Initialize Environment ---
env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
print("Habitat environment created.")

# --- Collect Dataset ---
step_id = 0
dataset = []
for i in tqdm(range(scenes_size), desc="Processing scenes"):
    special_scene_id = get_scene_path(env.scene_subfolder, dataset_name, eval_id=i)
    print(f"Loading scene {i}: {special_scene_id}")
    for j in range(max_scene_instance):
        obs, info = env.reset(seed=seed, is_unique=True, sync_info=None, special_scene_id=special_scene_id)
        image_width, image_height = obs.size
        action_space = env.action_space
        sim = env.sim
        agent = sim.get_agent(0)
        # Prepare prompt for the model
        prompt = build_text_obs([info])[0]

        thought_score = pre_max_score = info["conf_score"]
        thought_bbox = info["bbox_vg"]
        for k in range(max_step_length):
            pre_agent_state = agent.get_state()
            scores = []
            bboxs = []
            # Get action from the Rule
            for action in action_space[:-1]: # Exclude "stop"
                obs_sim = sim.step(action)
                obs_pil = vut.observation_to_image(obs_sim["rgb"], "color").convert("RGB")

                grounding_response = call_grounding_from_pil(obs_pil, info["task_prompt"])
                scores.append(grounding_response["score"])
                bboxs.append(grounding_response["bbox"])

                agent.set_state(pre_agent_state)
            max_score = max(scores)
            
            if max_score > pre_max_score:
                pre_max_score = max_score
                
                max_index = scores.index(max_score)
                action = action_space[max_index]
                action_index = max_index
                bbox = bboxs[max_index]
            else:
                action = "stop"
                action_index = len(action_space) - 1
            
            thoughts = get_CoT_label(image_width, image_height, thought_bbox, thought_score, action)
            entry, image_name = json_entry(step_id, prompt, thoughts, action)
            dataset.append(entry)
            obs.save(os.path.join(IMAGE_FOLDER, image_name))

            step_id = step_id + 1
            # Take a step in the environment
            obs, reward, done, info = env.step(action_index, True)
            if done:
                break

            prompt = build_text_obs([info])[0]
            thought_bbox = bbox
            thought_score = pre_max_score

print("\nCollection loop completed.")

with open(FILE_NAME, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
print(f"数据已保存到 {FILE_NAME}")