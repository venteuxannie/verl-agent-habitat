import os
os.environ["HABITAT_SIM_LOG"] = "quiet"
import numpy as np
import cv2
import json
import magnum as mn
from typing import List, Union
from pycocotools import mask as mask_utils

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
    get_hm3d_scene_folder,
    load_hm3d_valid_objects,
)
import ray
from .habitat_IoU_utils import (
    HabitatIoUWorker,
    HM3DIoUWorker,
    agent_state_to_serializable,
    magnum_quat_to_np,
    magnum_vec3_to_np,
)
from .habitat_3dbox_utils import transform_object_aabb_to_camera_obb
from .reward_function import reward_function
from .third_party import call_caption_from_pil, call_caption_and_task_description, draw_box_on_image
from .constants import hm3d_test, replica_cad_grounding_categories, ReplicaCAD_to_DETR, replica_cad_detect_categories, REPLICACAD_TRAINING_SCENES, HM3D_TRAINING_SCENES
hm3d_cat = hm3d_test

class CreateHabitatEnv:
    """
    Factory function to create a Habitat environment instance.
    This function can be extended to support different types of Habitat environments.
    """
    def __init__(self, seed, dataset_name, scenes_size=10, max_scene_instance=100, max_step_length=10, alpha_conf=0.5):
        self.seed = seed
        self.dataset_name = dataset_name
        self.alpha_conf = alpha_conf
        if self.dataset_name == "HM3D":
            self.task_categories = list(hm3d_cat.keys())
            self.scene_subfolder = HM3D_TRAINING_SCENES
        elif self.dataset_name == "ReplicaCAD":
            self.task_categories = replica_cad_grounding_categories
            self.scene_subfolder = REPLICACAD_TRAINING_SCENES
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        # self.scene_subfolder = get_dataset_subfolder(dataset_name)[:scenes_size]
        self.max_scene_instance = max_scene_instance
        self.max_step_length = max_step_length
        # Scene
        self.sim = None
        self.scene_instance_counter = -1
        # Task
        self.step_counter = -1
        self.target_category = None # category in scene
        self.task_list = ["grounding", "segment", "3d-box"]
        self.task_type = None # "grounding", "segment", or "3d-box"
        self.task_prompt = None # Short caption describing the target object (from VLM)
        self.task_description = None # Natural language task description for the agent (from VLM)
        self.semantic_id = None # semantic_id in scene
        self.instance_id = None # instance_id in scene
        # Action space
        self.action_space = ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']
        # Success evaluator
        self.init_conf_score = 0.0
        # IoU worker - dataset specific
        if self.dataset_name == "ReplicaCAD":
            self.iou_worker = HabitatIoUWorker.remote(self.dataset_name, True)
        elif self.dataset_name == "HM3D":
            self.hm3d_iou_worker = HM3DIoUWorker.remote()
            self.scene_folder = None  # Will be set when scene is loaded
            self.hm3d_semantic_objects = None  # Cache for semantic_objects {semantic_id: (center, sizes, category)}
            self.hm3d_valid_objects = None  # Cache for valid (preprocessed) objects info
        self.target_obj_handle = None
        self.target_obj_rotation_np = None
        self.target_obj_translation_np = None
        ############################ eval only ################################
        self.last_scene_id = None
        self.init_pred = None       # NOTE: init_pred is initialized in step() function rather than reset() function.
        #######################################################################
    
    def _extract_semantic_objects(self):
        """
        Extract semantic objects from the current simulator's semantic_scene.
        This avoids redundant scene loading in HM3DUnoccludedMaskGenerator.
        
        Returns:
            dict: {semantic_id: (center, sizes, category)} for objects with valid AABBs
        """
        if self.dataset_name != "HM3D" or self.sim is None:
            return None
        
        semantic_objects = {}
        semantic_scene = self.sim.semantic_scene
        for obj in semantic_scene.objects:
            if obj and obj.category and obj.aabb.sizes[0] > 0:
                center = np.array(obj.aabb.center)
                sizes = np.array(obj.aabb.sizes)
                category = obj.category.name()
                semantic_objects[obj.semantic_id] = (center, sizes, category)
        
        return semantic_objects
        

    def step(self, pred_task_type, pred_task_prompt, action_index, is_valid_action):
        """Execute a step in the environment"""
        # Check Task Type
        is_correct_task_type = pred_task_type == self.task_type
        
        # Use pred_task_prompt if valid, otherwise fall back to self.task_prompt
        # This allows the model to use its own understanding of the target object
        effective_task_prompt = pred_task_prompt if pred_task_prompt else self.task_prompt
        
        # Tool routing: Variable initialization is transferred to the step() function.
        if self.step_counter == 0:
            obs = self.sim.get_sensor_observations()
            _, _, self.init_conf_score, _, _, self.init_pred, self.pre_phi = reward_function(is_correct_task_type, pred_task_type, obs, effective_task_prompt, 0, True, "move_forward", self.step_counter, self.sim.get_agent(0).get_state(), alpha_conf=self.alpha_conf)
        # Step counter
        self.step_counter += 1

        # sim.step
        action = self.action_space[action_index]
        if action != "stop":
            obs = self.sim.step(action)
        else:
            obs = self.sim.get_sensor_observations()
        
        # obs_pil
        self.obs_pil = vut.observation_to_image(obs["rgb"], "color").convert("RGB")
        gt_dict = None
        # # NOTE: only for debug, more efficient without this line
        # _, gt_dict = self.get_gt()
        # only detect task need to update the task_prompt
        if self.task_type == "detect":
            _, gt_dict = self.get_gt()
            self.task_prompt = self.add_random_perturbation(gt_dict["bbox_gt"])
            self.obs_pil = draw_box_on_image(self.obs_pil, self.task_prompt[0], self.task_prompt[2], self.task_prompt[1], self.task_prompt[3])
        # reward (use effective_task_prompt for reward calculation)
        reward, format_score, self.pre_conf_score, img_vg, step_penalty, pred, self.pre_phi = reward_function(is_correct_task_type, pred_task_type, obs, effective_task_prompt, self.pre_phi, is_valid_action, action, self.step_counter, self.sim.get_agent(0).get_state(), alpha_conf=self.alpha_conf)
        # info
        info = {
            "task_type": self.task_type,
            "target_category": self.target_category,
            "semantic_id": self.semantic_id,
            "instance_id": self.instance_id,
            "obj_handle": self.target_obj_handle,
            "obj_rotation": self.target_obj_rotation_np,
            "obj_translation": self.target_obj_translation_np,
            "gt": gt_dict,
            "pred": pred,
            "task_prompt": self.task_prompt,
            "effective_task_prompt": effective_task_prompt,  # The task_prompt actually used for reward calculation
            "task_description": self.task_description,
            "conf_score": self.pre_conf_score,
            "phi": self.pre_phi,
            "scene_id": self.scene_id,
            "agent_pos": {},
            "won": 0,
        }
        # is_done
        if action == "stop" or self.step_counter == self.max_step_length:
            done = True
            # Get complete ground truth (bbox + mask)
            _, info["gt"] = self.get_gt()

            # Check if the task is won (for "success_evaluator")
            if self.pre_conf_score >= self.init_conf_score:
                info["won"] = 1
            else:
                info["won"] = 0
        else:
            done = False

        return self.obs_pil, reward, done, info

    def reset(self, seed=0, is_unique=True, sync_info=None, special_scene_id=None, task_type=None):
        """Reset the environment"""
        # Set task_type First!
        if task_type is None:
            self.task_type = random.choice(self.task_list)
        else:
            self.task_type = task_type
        
        # Reset the task only
        if -1 < self.scene_instance_counter < self.max_scene_instance-1:
            # Target-object
            if is_unique:
                # Get valid semantic IDs for HM3D (already loaded when scene was created)
                valid_ids = None
                if self.dataset_name == "HM3D" and self.hm3d_valid_objects is not None:
                    valid_ids = self.hm3d_valid_objects.get("valid_semantic_ids", None)
                self.target_category, self.semantic_id, self.obs_pil, bbox_gt_visible, agent_pos = get_an_instance(
                    dataset_name=self.dataset_name,
                    sim=self.sim,
                    target_categories=self.task_categories,
                    valid_semantic_ids=valid_ids
                )
                if self.dataset_name == "ReplicaCAD": # NOTE: for ReplicaCAD, the semantic_id is the class_id, not the instance_id
                    # NOTE: self.target_obj_handle, self.target_obj_rotation, self.target_obj_translation are initialized in CAD_get_bbox_gt
                    gt_dict, self.instance_id = self.CAD_get_bbox_gt(self.semantic_id)
                    bbox_gt = gt_dict["bbox_gt"]
                    # Store the complete gt_dict for later use
                    self._initial_gt_dict = gt_dict
                elif self.dataset_name == "HM3D":
                    # Get unoccluded mask, bbox, and 3D box for HM3D
                    gt_dict, self.instance_id = self.HM3D_get_bbox_gt(self.semantic_id)
                    bbox_gt = gt_dict["bbox_gt"]
                    self._initial_gt_dict = gt_dict
            else:
                self.task_type = sync_info["task_type"]
                # Support both old format (bbox_gt) and new format (gt dict)
                if "gt" in sync_info:
                    bbox_gt = sync_info["gt"]["bbox_gt"]
                elif "bbox_gt" in sync_info:
                    bbox_gt = sync_info["bbox_gt"]
                else:
                    bbox_gt = None
                self.target_category, self.semantic_id, self.instance_id, agent_pos = sync_info["target_category"], sync_info["semantic_id"], sync_info["instance_id"], sync_info["agent_pos"]
                self.target_obj_handle, self.target_obj_rotation_np, self.target_obj_translation_np = sync_info["obj_handle"], sync_info["obj_rotation"], sync_info["obj_translation"]
                self.obs_pil = sync_agent_state(self.sim, agent_pos)
            self.step_counter = 0
            self.scene_instance_counter += 1
        # Reset the scene and task
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
                self.sim = create_hm3d_simulator(self.dataset_name, self.scene_id, enable_semantic=True, enable_depth=True)
            else:
                self.scene_id = sync_info["scene_id"]
                self.sim = create_hm3d_simulator(self.dataset_name, self.scene_id, enable_semantic=True, enable_depth=True)
            # Set scene_folder and extract semantic_objects for HM3D
            if self.dataset_name == "HM3D":
                self.scene_folder = get_hm3d_scene_folder(self.scene_id)
                self.hm3d_semantic_objects = self._extract_semantic_objects()
                self.hm3d_valid_objects = load_hm3d_valid_objects(self.scene_folder)
            # Task
            self.step_counter = 0
            if is_unique:
                # Get valid semantic IDs for HM3D
                valid_ids = self.hm3d_valid_objects.get("valid_semantic_ids", None) if self.dataset_name == "HM3D" else None
                self.target_category, self.semantic_id, self.obs_pil, bbox_gt_visible, agent_pos = get_an_instance(
                    dataset_name=self.dataset_name,
                    sim=self.sim,
                    target_categories=self.task_categories,
                    valid_semantic_ids=valid_ids
                )
                if self.dataset_name == "ReplicaCAD": # NOTE: for ReplicaCAD, the semantic_id is the class_id, not the instance_id
                    # NOTE: self.target_obj_handle, self.target_obj_rotation, self.target_obj_translation are initialized in CAD_get_bbox_gt
                    gt_dict, self.instance_id = self.CAD_get_bbox_gt(self.semantic_id)
                    bbox_gt = gt_dict["bbox_gt"]
                    # Store the complete gt_dict for later use
                    self._initial_gt_dict = gt_dict
                elif self.dataset_name == "HM3D":
                    # Get unoccluded mask, bbox, and 3D box for HM3D
                    gt_dict, self.instance_id = self.HM3D_get_bbox_gt(self.semantic_id)
                    bbox_gt = gt_dict["bbox_gt"]
                    self._initial_gt_dict = gt_dict
            else:
                self.task_type = sync_info["task_type"]
                # Support both old format (bbox_gt) and new format (gt dict)
                if "gt" in sync_info:
                    bbox_gt = sync_info["gt"]["bbox_gt"]
                elif "bbox_gt" in sync_info:
                    bbox_gt = sync_info["bbox_gt"]
                else:
                    bbox_gt = None
                self.target_category, self.semantic_id, self.instance_id, agent_pos = sync_info["target_category"], sync_info["semantic_id"], sync_info["instance_id"], sync_info["agent_pos"]
                self.target_obj_handle, self.target_obj_rotation_np, self.target_obj_translation_np = sync_info["obj_handle"], sync_info["obj_rotation"], sync_info["obj_translation"]
                self.obs_pil = sync_agent_state(self.sim, agent_pos)
        # Task Prompt & Task Description
        if is_unique:
            if self.task_type in ["grounding", "segment", "3d-box"]:
                xmin, xmax, ymin, ymax = bbox_gt
                # Call VLM service to generate both caption (task_prompt) and natural language task_description
                
                # # NOTE:debug
                # response = {"task_prompt": "a red chair", "task_description": "find the red chair"}
                response = call_caption_and_task_description(
                    self.obs_pil, xmin, ymin, xmax, ymax, 
                    self.target_category, self.task_type
                )
                self.task_prompt = response["task_prompt"]
                self.task_description = response["task_description"]
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
        else:
            self.task_prompt = sync_info["task_prompt"]
            self.task_description = sync_info.get("task_description", self.task_prompt)
        # Init conf_score & phi
        pred = None
        self.pre_conf_score = None
        self.pre_phi = None

        # Prepare gt_dict for info
        # Use the stored _initial_gt_dict if available (from CAD_get_bbox_gt)
        if hasattr(self, '_initial_gt_dict') and self._initial_gt_dict is not None:
            gt_dict = self._initial_gt_dict
        else:
            # Fallback for sync_info or other cases
            gt_dict = {
                "bbox_gt": bbox_gt,
                "mask_gt": None
            }

        info = {
            "task_type": self.task_type,
            "target_category": self.target_category,
            "semantic_id": self.semantic_id,
            "instance_id": self.instance_id,
            "obj_handle": self.target_obj_handle,
            "obj_rotation": self.target_obj_rotation_np,
            "obj_translation": self.target_obj_translation_np,
            "gt": gt_dict,
            "pred": pred,
            "task_prompt": self.task_prompt,
            "effective_task_prompt": None,  # The task_prompt actually used for reward calculation
            "task_description": self.task_description,
            "conf_score": self.pre_conf_score,
            "phi": self.pre_phi,
            "scene_id": self.scene_id,
            "agent_pos": agent_pos,
        }

        return self.obs_pil, info
    
    ################### For evaluation only ###################
    ###########################################################
    def reset_eval(self, sync_info=None):
        """Reset the environment without the limit of max_scene_instance"""
        """Only depends on sync_info and special_scene_id"""
        # NOTE: sync_info may not contain "task_type"
        self.task_type = sync_info.get("task_type", "grounding")
        self.scene_id = sync_info["scene_id"]
        
        # Support both old format (bbox_gt) and new format (gt dict)
        if "gt" in sync_info:
            gt_dict = sync_info["gt"]
            bbox_gt = gt_dict["bbox_gt"]
        elif "bbox_gt" in sync_info:
            # Backward compatibility
            bbox_gt = sync_info["bbox_gt"]
            gt_dict = {"bbox_gt": bbox_gt, "mask_gt": None}
        else:
            bbox_gt = None
            gt_dict = {"bbox_gt": None, "mask_gt": None}
        
        # Reset the task only
        if self.scene_id == self.last_scene_id and self.scene_id != "NONE":
            self.target_category, self.semantic_id, self.instance_id, agent_pos = sync_info["target_category"], sync_info["semantic_id"], sync_info["instance_id"], sync_info["agent_pos"]
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
            self.sim = create_hm3d_simulator(self.dataset_name, self.scene_id, enable_semantic=True, enable_depth=True)
            # Set scene_folder and extract semantic_objects for HM3D
            if self.dataset_name == "HM3D":
                self.scene_folder = get_hm3d_scene_folder(self.scene_id)
                self.hm3d_semantic_objects = self._extract_semantic_objects()
                self.hm3d_valid_objects = load_hm3d_valid_objects(self.scene_folder)
            # Task
            self.step_counter = 0
            self.target_category, self.semantic_id, self.instance_id, agent_pos = sync_info["target_category"], sync_info["semantic_id"], sync_info["instance_id"], sync_info["agent_pos"]
            self.target_obj_handle, self.target_obj_rotation_np, self.target_obj_translation_np = sync_info["obj_handle"], sync_info["obj_rotation"], sync_info["obj_translation"]
            self.obs_pil = sync_agent_state(self.sim, agent_pos)
        # Task Prompt & Task Description
        self.task_prompt = sync_info["task_prompt"]
        self.task_description = sync_info.get("task_description", self.task_prompt)

        pred = None
        self.pre_conf_score = None
        self.pre_phi = None

        info = {
            "task_type": self.task_type,
            "target_category": self.target_category,
            "semantic_id": self.semantic_id,
            "instance_id": self.instance_id,
            "obj_handle": self.target_obj_handle,
            "obj_rotation": self.target_obj_rotation_np,
            "obj_translation": self.target_obj_translation_np,
            "gt": gt_dict,
            "pred": pred,
            "task_prompt": self.task_prompt,
            "effective_task_prompt": None,  # The task_prompt actually used for reward calculation
            "task_description": self.task_description,
            "conf_score": self.pre_conf_score,
            "phi": self.pre_phi,
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
            future = self.iou_worker.reset.remote(
                agent_state_to_serializable(agent_state),
                obj_handle,
                magnum_quat_to_np(obj_rotation),
                magnum_vec3_to_np(obj_translation)
            )

            obs_empty = ray.get(future)

            # check whether the object is visible in the empty scene
            semantic = obs_empty["semantic"]
            unique_ids = np.unique(semantic)
            if len(unique_ids) > 1: # 0 & semantic_id
                target_obj_id = obj_id
                target_obs_empty = obs_empty

                target_obj_handle = obj_handle
                target_obj_rotation = obj_rotation
                target_obj_translation = obj_translation
                break

        ################# (unoccluded) bbox_gt & mask_gt #################
        semantic_np = np.array(target_obs_empty["semantic"])

        # get obj_mask
        mask = (semantic_np == semantic_id).astype(np.uint8)
        
        # 计算边界框:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bbox_gt = (xmin, xmax, ymin, ymax)
        
        # 编码 mask 为 RLE 格式
        mask_rle = mask_utils.encode(np.asfortranarray(mask))
        # 转换 bytes 为 str 以便 JSON 序列化
        if isinstance(mask_rle['counts'], bytes):
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
        
        self.target_obj_handle = target_obj_handle
        self.target_obj_rotation_np = magnum_quat_to_np(target_obj_rotation)
        self.target_obj_translation_np = magnum_vec3_to_np(target_obj_translation)

        ########### 3D bbox GT ###########
        obj = rigid_obj_mgr.get_object_by_id(target_obj_id)
        bbox_3d_gt = transform_object_aabb_to_camera_obb(obj, agent_state, "rgb")
        
        ########### 返回 gt_dict 包含 bbox、mask 和 3D bbox ###########
        gt_dict = {
            "bbox_gt": bbox_gt,
            "mask_gt": mask_rle,
            "bbox_3d_gt": bbox_3d_gt
        }
        
        return gt_dict, target_obj_id
    
    
    def HM3D_get_bbox_gt(self, semantic_id):
        """
        Get unoccluded bbox, mask, and 3D box for a HM3D object.
        
        For HM3D, objects are part of the scene mesh. We extract the object mesh
        based on vertex colors and render it in isolation to get unoccluded views.
        
        Args:
            semantic_id: The semantic ID of the target object
            
        Returns:
            gt_dict: Dictionary containing:
                - bbox_gt: (xmin, xmax, ymin, ymax) bounding box
                - mask_gt: RLE format segmentation mask
                - bbox_3d_gt: 3D bounding box information
            instance_id: Same as semantic_id for HM3D (no separate instance tracking)
        """
        agent_state = self.sim.get_agent(0).get_state()
        
        # Call HM3D IoU worker to get unoccluded mask
        # Pass pre-extracted semantic_objects to avoid redundant scene loading
        future = self.hm3d_iou_worker.reset.remote(
            agent_state_to_serializable(agent_state),
            self.scene_folder,
            semantic_id,
            self.hm3d_semantic_objects
        )
        
        result = ray.get(future)
        
        # Get occluded mask from current scene for merging
        obs_current = self.sim.get_sensor_observations()
        semantic_current = np.array(obs_current["semantic"])
        occluded_mask = (semantic_current == semantic_id).astype(np.uint8)
        
        # Check if object is visible in unoccluded view
        obs = result["obs"]
        semantic_np = np.array(obs["semantic"])
        unique_ids = np.unique(semantic_np)
        
        if len(unique_ids) <= 1:  # Only 0 (background) in unoccluded view
            # Object not visible from this viewpoint - use only occluded mask
            if np.any(occluded_mask):
                rows = np.any(occluded_mask, axis=1)
                cols = np.any(occluded_mask, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                bbox_gt = (xmin, xmax, ymin, ymax)
                mask_rle = mask_utils.encode(np.asfortranarray(occluded_mask))
                if isinstance(mask_rle['counts'], bytes):
                    mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            else:
                bbox_gt = (0, 0, 0, 0)
                empty_mask = np.zeros_like(semantic_current, dtype=np.uint8)
                mask_rle = mask_utils.encode(np.asfortranarray(empty_mask))
                if isinstance(mask_rle['counts'], bytes):
                    mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            
            gt_dict = {
                "bbox_gt": bbox_gt,
                "mask_gt": mask_rle,
                "bbox_3d_gt": result.get("bbox_3d_gt", None)
            }
            return gt_dict, semantic_id
        
        # Decode unoccluded mask from IoU worker result
        unoccluded_mask_rle = result["mask_gt"]
        if isinstance(unoccluded_mask_rle['counts'], str):
            unoccluded_mask_rle['counts'] = unoccluded_mask_rle['counts'].encode('utf-8')
        unoccluded_mask = mask_utils.decode(unoccluded_mask_rle).astype(np.uint8)
        
        # Merge unoccluded mask and occluded mask (union) to fill holes
        merged_mask = np.logical_or(unoccluded_mask, occluded_mask).astype(np.uint8)
        
        # Recalculate bbox based on merged mask
        if np.any(merged_mask):
            rows = np.any(merged_mask, axis=1)
            cols = np.any(merged_mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            bbox_gt = (xmin, xmax, ymin, ymax)
        else:
            bbox_gt = (0, 0, 0, 0)
        
        # Encode merged mask to RLE format
        mask_rle = mask_utils.encode(np.asfortranarray(merged_mask))
        if isinstance(mask_rle['counts'], bytes):
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
        
        gt_dict = {
            "bbox_gt": bbox_gt,
            "mask_gt": mask_rle,
            "bbox_3d_gt": result["bbox_3d_gt"]
        }
        
        return gt_dict, semantic_id



    def get_gt(self):
        """
        Get the current ground truth information including bbox and mask.
        
        Returns:
            rgb_pil: PIL Image of the observation
            gt_dict: Dictionary containing:
                - bbox_gt: (xmin, xmax, ymin, ymax) bounding box
                - mask_gt: RLE format segmentation mask
        """
        obs_current = self.sim.get_sensor_observations()
        rgb_pil = vut.observation_to_image(obs_current["rgb"], "color").convert("RGB")
        
        if self.dataset_name == "ReplicaCAD":
            agent_state = self.sim.get_agent(0).get_state()
            future = self.iou_worker.reset.remote(
                agent_state_to_serializable(agent_state),
                self.target_obj_handle,
                self.target_obj_rotation_np,
                self.target_obj_translation_np
            )
            obs_empty = ray.get(future)

            # Get semantic segmentation
            semantic_np = np.array(obs_empty["semantic"])

            # Create binary mask for the target object
            obj_mask = (semantic_np > 0).astype(np.uint8)
            
            # Calculate bounding box
            if np.unique(obj_mask).size == 1:
                # No object visible
                bbox_gt = (0, 0, 0, 0)
                # Create empty mask
                empty_mask = np.zeros_like(obj_mask, dtype=np.uint8)
                mask_rle = mask_utils.encode(np.asfortranarray(empty_mask))
            else:
                rows = np.any(obj_mask, axis=1)
                cols = np.any(obj_mask, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                bbox_gt = (xmin, xmax, ymin, ymax)
                
                # Encode mask to RLE format
                mask_rle = mask_utils.encode(np.asfortranarray(obj_mask))
            
            # Convert bytes to str for JSON serialization
            if isinstance(mask_rle['counts'], bytes):
                mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            
            ########### 3D bbox GT ###########
            rigid_obj_mgr = self.sim.get_rigid_object_manager()
            obj = rigid_obj_mgr.get_object_by_id(self.instance_id)
            bbox_3d_gt = transform_object_aabb_to_camera_obb(obj, agent_state, "rgb")

            gt_dict = {
                "bbox_gt": bbox_gt,
                "mask_gt": mask_rle,
                "bbox_3d_gt": bbox_3d_gt
            }
        
        elif self.dataset_name == "HM3D":
            # Use HM3D IoU worker for unoccluded mask
            # Pass pre-extracted semantic_objects to avoid redundant scene loading
            agent_state = self.sim.get_agent(0).get_state()
            future = self.hm3d_iou_worker.reset.remote(
                agent_state_to_serializable(agent_state),
                self.scene_folder,
                self.semantic_id,
                self.hm3d_semantic_objects
            )
            result = ray.get(future)
            
            # Get occluded mask from current scene
            semantic_current = np.array(obs_current["semantic"])
            occluded_mask = (semantic_current == self.semantic_id).astype(np.uint8)
            
            # Decode unoccluded mask from IoU worker result
            unoccluded_mask_rle = result["mask_gt"]
            if isinstance(unoccluded_mask_rle['counts'], str):
                unoccluded_mask_rle['counts'] = unoccluded_mask_rle['counts'].encode('utf-8')
            unoccluded_mask = mask_utils.decode(unoccluded_mask_rle).astype(np.uint8)
            
            # Merge unoccluded mask and occluded mask (union) to fill holes
            merged_mask = np.logical_or(unoccluded_mask, occluded_mask).astype(np.uint8)
            
            # Recalculate bbox based on merged mask
            if np.any(merged_mask):
                rows = np.any(merged_mask, axis=1)
                cols = np.any(merged_mask, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                bbox_gt = (xmin, xmax, ymin, ymax)
            else:
                bbox_gt = (0, 0, 0, 0)
            
            # Encode merged mask to RLE format
            mask_rle = mask_utils.encode(np.asfortranarray(merged_mask))
            if isinstance(mask_rle['counts'], bytes):
                mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            
            gt_dict = {
                "bbox_gt": bbox_gt,
                "mask_gt": mask_rle,
                "bbox_3d_gt": result["bbox_3d_gt"]
            }
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        return rgb_pil, gt_dict
    
    def get_gt_for_class(self):
        """
        Get ground truth for all instances of the target class (for class-detect task).
        逐个加载物体来获取每个实例的准确 bbox 和 mask。
        
        Returns:
            rgb_pil: PIL Image of the observation (from current view)
            gt_list: List of dictionaries, each containing:
                - bbox_gt: (x1, y1, x2, y2) bounding box
                - mask_gt: RLE format segmentation mask
                - instance_id: instance ID of this object
                - category: object category
        """
        # Get all objects of the same class (semantic_id)
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        obj_dict = rigid_obj_mgr.get_objects_by_handle_substring()
        
        # Find all objects with the same semantic_id
        class_obj_list = []
        for obj_id, obj in obj_dict.items():
            if obj.semantic_id == self.semantic_id:
                obj_info = {
                    'obj_handle': obj.creation_attributes.handle,
                    'obj_rotation_np': magnum_quat_to_np(obj.rotation),
                    'obj_translation_np': magnum_vec3_to_np(obj.translation),
                    'instance_id': obj.object_id,
                }
                class_obj_list.append(obj_info)
        
        if not class_obj_list:
            # No objects found, return empty
            return None, []
        
        # Get agent state
        agent_state = self.sim.get_agent(0).get_state()
        
        # 逐个加载物体获取 GT（reset_class 现在返回的是 obs_list）
        future = self.iou_worker.reset_class.remote(
            agent_state_to_serializable(agent_state),
            class_obj_list
        )
        obs_list = ray.get(future)
        
        # 获取当前视角的观察（用于过滤被遮挡的物体）
        obs_current = self.sim.get_sensor_observations()
        rgb_pil = vut.observation_to_image(obs_current["rgb"], "color").convert("RGB")
        semantic_current = np.array(obs_current["semantic"])
        
        gt_list = []
        
        # 处理每个物体的观察结果
        for obs_with_id in obs_list:
            obs_empty = obs_with_id['obs']
            instance_id = obs_with_id['instance_id']
            
            # Get semantic segmentation from empty scene
            semantic_np = np.array(obs_empty["semantic"])
            
            # 创建该物体的 mask（语义分割中非0的部分）
            obj_mask = (semantic_np > 0).astype(np.uint8)
            
            # 检查物体在空场景中是否可见
            if np.sum(obj_mask) == 0:
                continue  # 物体在空场景中不可见，跳过
            
            # Calculate bounding box from empty scene
            rows = np.any(obj_mask, axis=1)
            cols = np.any(obj_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                continue
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # Convert bbox format to (x1, y1, x2, y2) for detection
            bbox_gt = (xmin, ymin, xmax, ymax)
            
            # ========== 关键过滤步骤：检查在当前实际场景中是否可见 ==========
            # 在 bbox 范围内检查是否有该类别（semantic_id）的可见像素
            x1, y1, x2, y2 = bbox_gt
            # 确保坐标在图像范围内
            H, W = semantic_current.shape
            x1_clipped = max(0, min(W - 1, x1))
            x2_clipped = max(0, min(W - 1, x2))
            y1_clipped = max(0, min(H - 1, y1))
            y2_clipped = max(0, min(H - 1, y2))
            
            # 提取 bbox 区域的语义分割
            bbox_region = semantic_current[y1_clipped:y2_clipped+1, x1_clipped:x2_clipped+1]
            
            # 检查该区域内是否有该类别的像素
            visible_pixels = np.sum(bbox_region == self.semantic_id)
            
            if visible_pixels == 0:
                # 该物体在当前视角下完全被遮挡，剔除
                continue
            # ==============================================================
            
            # Encode mask to RLE format
            mask_rle = mask_utils.encode(np.asfortranarray(obj_mask))
            
            # Convert bytes to str for JSON serialization
            if isinstance(mask_rle['counts'], bytes):
                mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            
            gt_dict = {
                "bbox_gt": bbox_gt,
                "mask_gt": mask_rle,
                "instance_id": int(instance_id),
                "category": self.target_category,
                "visible_pixels": int(visible_pixels)  # 可选：记录可见像素数
            }
            
            gt_list.append(gt_dict)
        
        return rgb_pil, gt_list

    def add_random_perturbation(self, bbox):
        # 给task_prompt增加一个小的随机偏移量
        x1, x2, y1, y2 = bbox
        # 控制偏移范围：最多在±5% bbox宽高；同时至少取整, 最小=±1像素
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        max_x_offset = max(1, int(w * 0.05))
        max_y_offset = max(1, int(h * 0.05))
        ox1 = random.randint(-max_x_offset, max_x_offset)
        ox2 = random.randint(-max_x_offset, max_x_offset)
        oy1 = random.randint(-max_y_offset, max_y_offset)
        oy2 = random.randint(-max_y_offset, max_y_offset)
        # 偏移后裁剪到合法范围（不能逆序，不能超出图像范围）
        W, H = self.obs_pil.size if hasattr(self, "obs_pil") else (640, 480)
        nx1 = max(0, min(W-1, x1 + ox1))
        nx2 = max(0, min(W-1, x2 + ox2))
        ny1 = max(0, min(H-1, y1 + oy1))
        ny2 = max(0, min(H-1, y2 + oy2))
        # 确保bbox顺序正确
        nx1, nx2 = min(nx1, nx2), max(nx1, nx2)
        ny1, ny2 = min(ny1, ny2), max(ny1, ny2)
        return (nx1, nx2, ny1, ny2)

    def get_task_type(self):
        return self.task_type

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