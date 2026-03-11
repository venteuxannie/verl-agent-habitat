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

import math
import random
import torch

from .habitat_utils import (
    create_hm3d_simulator,
    get_dataset_subfolder,
    get_random_scene_path,
    get_gt_bbox,
    get_an_instance,
    sync_agent_state,
)
# import ray
from .habitat_IoU_utils import (
    HabitatIoUWorker,
    agent_state_to_serializable,
    magnum_quat_to_np,
    magnum_vec3_to_np,
)
from .reward_function import reward_function
from .third_party import call_grounding_from_pil, call_caption_from_pil
from .constants import hm3d_test, replica_cad_cat_test
hm3d_cat = hm3d_test
replica_cad_cat = replica_cad_cat_test

class CreateHabitatEnv:
    """
    Factory function to create a Habitat environment instance.
    This function can be extended to support different types of Habitat environments.
    """
    def __init__(self, seed, dataset_name, scenes_size=10, max_scene_instance=100, max_step_length=10):
        self.seed = seed
        self.dataset_name = dataset_name
        if self.dataset_name == "HM3D":
            self.task_categories = list(hm3d_cat.keys())
        elif self.dataset_name == "ReplicaCAD":
            self.task_categories = replica_cad_cat
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        self.scene_subfolder = get_dataset_subfolder(dataset_name)[:scenes_size]
        self.max_scene_instance = max_scene_instance
        self.max_step_length = max_step_length
        # Scene
        self.sim = None
        self.scene_instance_counter = -1
        # Task
        self.step_counter = -1
        # Action space
        self.action_space = ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']
        # Success evaluator
        self.init_conf_score = 0.0
        # IoU worker
        self.iou_worker = HabitatIoUWorker(self.dataset_name, True)
        self.target_obj_handle = None
        self.target_obj_rotation_np = None
        self.target_obj_translation_np = None
        ############################ eval only ################################
        self.last_scene_id = None
        #######################################################################
        

    def step(self, action_index, is_valid_action):
        """Execute a step in the environment"""
        self.step_counter += 1

        # sim.step
        action = self.action_space[action_index]
        if action != "stop":
            obs = self.sim.step(action)
        else:
            obs = self.sim.get_sensor_observations()
        
        # obs_pil
        self.obs_pil = vut.observation_to_image(obs["rgb"], "color")
        # reward
        reward, format_score, self.pre_conf_score, img_vg, step_penalty, bbox_vg = reward_function(self.obs_pil, self.task_prompt, self.pre_conf_score, is_valid_action, action, self.step_counter)
        # info
        info = {
            "target_category": self.target_category,
            "instance_id": self.instance_id,
            "obj_handle": self.target_obj_handle,
            "obj_rotation": self.target_obj_rotation_np,
            "obj_translation": self.target_obj_translation_np,
            "bbox_gt": None,
            "bbox_vg": bbox_vg,
            "task_prompt": self.task_prompt,
            "conf_score": self.pre_conf_score,
            "scene_id": self.scene_id,
            "agent_pos": {},
            "won": 0,
        }
        # is_done
        if action == "stop" or self.step_counter > self.max_step_length:
            done = True
            # bbox_gt
            if self.dataset_name == "ReplicaCAD":
                 info["bbox_gt"] = self.get_bbox_gt()
            elif self.dataset_name == "HM3D":
                _, info["bbox_gt"], _ = get_gt_bbox(obs, self.instance_id)
            # Check if the task is won (for "success_evaluator")
            if self.pre_conf_score >= self.init_conf_score:
                info["won"] = 1
            else:
                info["won"] = 0
        else:
            done = False

        return self.obs_pil, reward, done, info

    def reset(self, seed=0, is_unique=True, sync_info=None, special_scene_id=None):
        """Reset the environment"""
        # Reset the task only
        if -1 < self.scene_instance_counter < self.max_scene_instance-1:
            if is_unique:
                self.target_category, self.instance_id, self.obs_pil, bbox_gt, agent_pos = get_an_instance(
                    dataset_name=self.dataset_name,
                    sim=self.sim,
                    target_categories=self.task_categories
                )
                if self.dataset_name == "ReplicaCAD": # NOTE: for ReplicaCAD, the semantic_id is the class_id, not the instance_id
                    # NOTE: self.target_obj_handle, self.target_obj_rotation, self.target_obj_translation are initialized in CAD_get_bbox_gt
                    bbox_gt, self.instance_id = self.CAD_get_bbox_gt(self.instance_id)
            else:
                self.target_category, self.instance_id, bbox_gt, agent_pos = sync_info["target_category"], sync_info["instance_id"], sync_info["bbox_gt"], sync_info["agent_pos"]
                self.target_obj_handle, self.target_obj_rotation_np, self.target_obj_translation_np = sync_info["obj_handle"], sync_info["obj_rotation"], sync_info["obj_translation"]
                self.obs_pil = sync_agent_state(self.sim, agent_pos)
            self.step_counter = 0
            self.scene_instance_counter += 1
        # Rest the scene and task
        else:
            if self.scene_instance_counter != -1:
                self.sim.close()
            # Scene
            self.scene_instance_counter = 0
            if is_unique:
                if special_scene_id is not None:
                    self.scene_id = special_scene_id
                else:
                    self.scene_id = get_random_scene_path(self.scene_subfolder, self.dataset_name)
                self.sim = create_hm3d_simulator(self.dataset_name, self.scene_id, enable_semantic=True)
            else:
                self.scene_id = sync_info["scene_id"]
                self.sim = create_hm3d_simulator(self.dataset_name, self.scene_id, enable_semantic=True)
            # Task
            self.step_counter = 0
            if is_unique:
                self.target_category, self.instance_id, self.obs_pil, bbox_gt, agent_pos = get_an_instance(
                    dataset_name=self.dataset_name,
                    sim=self.sim,
                    target_categories=self.task_categories
                )
                if self.dataset_name == "ReplicaCAD": # NOTE: for ReplicaCAD, the semantic_id is the class_id, not the instance_id
                    # NOTE: self.target_obj_handle, self.target_obj_rotation, self.target_obj_translation are initialized in CAD_get_bbox_gt
                    bbox_gt, self.instance_id = self.CAD_get_bbox_gt(self.instance_id)
            else:
                self.target_category, self.instance_id, bbox_gt, agent_pos = sync_info["target_category"], sync_info["instance_id"], sync_info["bbox_gt"], sync_info["agent_pos"]
                self.target_obj_handle, self.target_obj_rotation_np, self.target_obj_translation_np = sync_info["obj_handle"], sync_info["obj_rotation"], sync_info["obj_translation"]
                self.obs_pil = sync_agent_state(self.sim, agent_pos)
        # Task Prompt
        if is_unique:
            xmin, xmax, ymin, ymax = bbox_gt #TODO：task_prompt=="useless"时，重新获取instance_id
            caption_response = call_caption_from_pil(self.obs_pil, xmin, ymin, xmax, ymax, self.target_category)
            self.task_prompt = caption_response["caption"]
        else:
            self.task_prompt = sync_info["task_prompt"]
        # init conf_score
        bbox_vg = None
        if is_unique:
            grounding_response = call_grounding_from_pil(self.obs_pil, self.task_prompt)
            self.pre_conf_score = self.init_conf_score = grounding_response["score"]
            bbox_vg = grounding_response["bbox"]
        else:
            self.pre_conf_score = self.init_conf_score = sync_info["conf_score"]

        info = {
            "target_category": self.target_category,
            "instance_id": self.instance_id,
            "obj_handle": self.target_obj_handle,
            "obj_rotation": self.target_obj_rotation_np,
            "obj_translation": self.target_obj_translation_np,
            "bbox_gt": bbox_gt,
            "bbox_vg": bbox_vg,
            "task_prompt": self.task_prompt,
            "conf_score": self.pre_conf_score,
            "scene_id": self.scene_id,
            "agent_pos": agent_pos,
        }

        return self.obs_pil, info
    
    ################### For evaluation only ###################
    ###########################################################
    def reset_eval(self, sync_info=None):
        """Reset the environment without the limit of max_scene_instance"""
        """Only depends on sync_info and special_scene_id"""
        self.scene_id = sync_info["scene_id"]
        # Reset the task only
        if self.scene_id == self.last_scene_id and self.scene_id != "NONE":
            self.target_category, self.instance_id, bbox_gt, agent_pos = sync_info["target_category"], sync_info["instance_id"], sync_info["bbox_gt"], sync_info["agent_pos"]
            self.target_obj_handle, self.target_obj_rotation_np, self.target_obj_translation_np = sync_info["obj_handle"], sync_info["obj_rotation"], sync_info["obj_translation"]
            self.obs_pil = sync_agent_state(self.sim, agent_pos)
            self.step_counter = 0
            self.scene_instance_counter += 1
        # Rest the scene and task
        else:
            self.last_scene_id = self.scene_id
            if self.sim != None:
                self.sim.close()
            # Scene
            self.scene_instance_counter = 0
            self.sim = create_hm3d_simulator(self.dataset_name, self.scene_id, enable_semantic=True)
            # Task
            self.step_counter = 0
            self.target_category, self.instance_id, bbox_gt, agent_pos = sync_info["target_category"], sync_info["instance_id"], sync_info["bbox_gt"], sync_info["agent_pos"]
            self.target_obj_handle, self.target_obj_rotation_np, self.target_obj_translation_np = sync_info["obj_handle"], sync_info["obj_rotation"], sync_info["obj_translation"]
            self.obs_pil = sync_agent_state(self.sim, agent_pos)
        # Task Prompt
        self.task_prompt = sync_info["task_prompt"]
        # init conf_score
        grounding_response = call_grounding_from_pil(self.obs_pil, self.task_prompt)
        self.pre_conf_score = self.init_conf_score = grounding_response["score"]
        bbox_vg = grounding_response["bbox"]

        info = {
            "target_category": self.target_category,
            "instance_id": self.instance_id,
            "obj_handle": self.target_obj_handle,
            "obj_rotation": self.target_obj_rotation_np,
            "obj_translation": self.target_obj_translation_np,
            "bbox_gt": bbox_gt,
            "bbox_vg": bbox_vg,
            "task_prompt": self.task_prompt,
            "conf_score": self.pre_conf_score,
            "scene_id": self.scene_id,
            "agent_pos": agent_pos,
        }

        return self.obs_pil, info
    ###########################################################

    def CAD_get_bbox_gt(self, semantic_id):
        ############## "instance_id(semantic_id)" -> object_id ##############
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        obj_dict = rigid_obj_mgr.get_objects_by_handle_substring()

        class_obj_list = []
        for obj_id, obj in obj_dict.items():
            if obj.semantic_id == semantic_id:
                class_obj_list.append(obj)

        agent_state = self.sim.get_agent(0).get_state()
        target_obj_id = None
        target_obs_empty = None

        target_obj_handle = None
        target_obj_rotation = None
        target_obj_translation = None
        for target_obj in class_obj_list:
            obj_id = target_obj.object_id
            obj_handle = target_obj.creation_attributes.handle
            obj_rotation = target_obj.rotation
            obj_translation = target_obj.translation

            # create an empty scene
            obs_empty = self.iou_worker.reset(
                agent_state_to_serializable(agent_state),
                obj_handle,
                magnum_quat_to_np(obj_rotation),
                magnum_vec3_to_np(obj_translation)
            )

            # obs_empty = ray.get(future)

            # check whether the object is visible in the empty scene
            semantic = obs_empty["semantic"]
            unique_ids = np.unique(semantic)
            if len(unique_ids) > 1:
                target_obj_id = obj_id
                target_obs_empty = obs_empty

                target_obj_handle = obj_handle
                target_obj_rotation = obj_rotation
                target_obj_translation = obj_translation
                break

        ################# (unoccluded) bbox_gt #################
        semantic_np = np.array(target_obs_empty["semantic"])

        # get obj_mask
        mask = (semantic_np == semantic_id)

        #NOTE: handle this situation
        pixel_area = int(np.sum(mask))
        if pixel_area == 0:
            return [0, 0, 0, 0], 0, target_obj_handle, target_obj_rotation, target_obj_translation
        
        # 计算边界框:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bbox_gt = (xmin, xmax, ymin, ymax)
        self.target_obj_handle = target_obj_handle
        self.target_obj_rotation_np = magnum_quat_to_np(target_obj_rotation)
        self.target_obj_translation_np = magnum_vec3_to_np(target_obj_translation)

        return bbox_gt, target_obj_id
    
    def get_bbox_gt(self):
        """Get the current ground truth bounding box"""
        agent_state = self.sim.get_agent(0).get_state()
        obs_empty = self.iou_worker.reset(
            agent_state_to_serializable(agent_state),
            self.target_obj_handle,
            self.target_obj_rotation_np,
            self.target_obj_translation_np
        )
        # obs_empty = ray.get(future)

        # ################# (unoccluded) bbox_gt #################
        semantic_np = np.array(obs_empty["semantic"])
        rgb_pil = vut.observation_to_image(obs_empty["rgb"], "color").convert("RGB")

        # get obj_mask
        mask = semantic_np

        # 计算边界框:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        return rgb_pil, (xmin, xmax, ymin, ymax)
    
    def get_rgb(self):
        """Get the current RGB observation"""
        obs = self.sim.get_sensor_observations()
        rgb = vut.observation_to_image(obs["rgb"], "color")
        return rgb

    def close(self):
        """
        显式关闭仿真器并释放相关资源。
        """
        if self.sim is not None:
            print("Closing the Habitat simulator.")
            self.sim.close()
            self.sim = None

    # =======================================================
    # 新增上下文管理器支持
    # =======================================================
    def __enter__(self):
        """当进入 with 语句块时被调用，返回 self 即可"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """当退出 with 语句块时被调用，在这里执行清理工作"""
        self.close()
    # =======================================================