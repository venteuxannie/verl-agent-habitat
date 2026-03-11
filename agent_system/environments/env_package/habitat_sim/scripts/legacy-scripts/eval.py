from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor

from .habitat_envs import CreateHabitatEnv
from agent_system.environments.prompts import *
from agent_system.multi_turn_rollout.utils import process_image
from functools import partial
from typing import List
import os

import random
def habitat_projection(text_actions: List[str], env_name):
    output_indices = []
    valids = []
    if env_name == 'habitat':
        action_list = ["move_forward", "turn_left", "turn_right", "look_up", "look_down", "stop"]
    elif env_name == 'gym_cards/NumberLine-v0':
        action_list = ["-", "+"]
    elif env_name == 'gym_cards/Blackjack-v0':
        action_list = ["stand", "hit"]
    elif env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    elif env_name == 'gym_cards/Points24-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    for string in text_actions:
        if not isinstance(string, str):
            # directly output a random action if the string is not a string
            output_indices.append(random.randint(0, len(action_list) - 1))
            valids.append(0)
            continue
        string = string.lower()
        action_index = string.find('"action":')
        # Extract everything after "action":
        string = string[action_index:]
        contained_actions = []
        # For the 'gym_cards/Points24-v0' environment, handle '10' separately
        if 'points' in env_name.lower() and '10' in string:
            contained_actions.append('10')
            string = string.replace('10', '')  # Remove '10' to prevent it from being counted as '1'
        # Find all actions that are contained in the string
        for action in action_list:
            if action in string:
                contained_actions.append(action)
        # Remove duplicates by converting to a set and back to a list
        contained_actions = list(set(contained_actions))
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            # Only one keyword from action_list is in the string
            output_indices.append(action_list.index(contained_actions[0]))
            valids.append(1)
        else:
            # The string contains none or multiple keywords, randomly select an index from action_list
            output_indices.append(random.randint(0, len(action_list) - 1))
            valids.append(0)
    return output_indices, valids

def load_model_and_processor(model_path: str):
    # Load the model in half-precision on the available device(s)
    if "Qwen2.5" in model_path:
        print("Loading Qwen2.5-VL model from", model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
    elif "Qwen2" in model_path:
        print("Loading Qwen2-VL model from", model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
    else:
        raise ValueError("Unsupported model path. Please provide a valid Qwen2 or Qwen2.5 model path.")
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor

def inference(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    
    return output_text if output_text else ""

def get_scene_path(subfolders, dataset_name, eval_id=0):
    data_path = "/data/tct/habitat/data"
    if dataset_name == "HM3D":
        dataset_path = os.path.join(data_path, "hm3d")
        scene = subfolders[eval_id % len(subfolders)]
        id = scene.split('/')[1].split('-')[1]
        scene_path = f"{scene}/{id}.basis.glb"
        scene_id = os.path.join(dataset_path, scene_path)   # glb文件路径 
    elif dataset_name == "ReplicaCAD":
        scene_name = subfolders[eval_id % len(subfolders)]
        scene_id = os.path.join(data_path, "replica_cad/configs/scenes", scene_name) # json文件路径
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return scene_id

def build_text_obs(infos: List[Dict]) -> List[str]:
    """
    This function builds the text observation (prompts) for the agent.
    """
    postprocess_text_obs = []
    for i in range(len(infos)):
        prompt = HABITAT_VISUAL_GROUNDING_COT_TEMPLATE.format(
            task_caption=infos[i]['task_prompt'],
            conf_score=infos[i]['conf_score']
        )

        postprocess_text_obs.append(prompt)
    return postprocess_text_obs

def evaluate():
    # model_path = "/data/tct/verl-agent/checkpoints/verl_agent_habitat_test/grpo_qwen2_vl_2b-230-test-bs_8-4cards/Qwen2-VL-2B-HM3D-10"
    model_path = "/data/tct/models/Qwen2-VL-2B-Instruct"
    model, processor = load_model_and_processor(model_path)
    print("Model and processor loaded successfully.")

    # dataset_name = "HM3D"
    dataset_name = "ReplicaCAD"
    seed = 0

    scenes_size=10
    max_scene_instance=1
    max_step_length=10
    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    projection_f = partial(habitat_projection, env_name='habitat')

    for i in range(scenes_size):
        special_scene_id = get_scene_path(env.scene_subfolder, dataset_name, eval_id=i)
        for j in range(max_scene_instance):
            obs, info = env.reset(seed=seed, is_unique=True, sync_info=None, special_scene_id=special_scene_id)
            
            for k in range(max_step_length):
                prompt = build_text_obs([info])[0].replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                print(f"Scene {i}: {info['scene_id']}")

                text_actions = inference(model, processor, process_image(obs), prompt)
                print(f"Text Actions: {text_actions}")
                actions, valids = projection_f(text_actions, env_name="habitat")
                obs, reward, done, info = env.step(actions[0], valids[0])

                if done:
                    break

if __name__ == "__main__":
    evaluate()
    print("Evaluation completed successfully.")