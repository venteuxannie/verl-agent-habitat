from .habitat_utils import create_hm3d_simulator
from .constants import TEMP_DIR
import ray

import numpy as np
from habitat_sim.agent import AgentState
from habitat_sim import SixDOFPose
import quaternion
from magnum import Vector3
from magnum import Quaternion, Vector3

# object utils
def magnum_quat_to_np(q):
    return np.array([q.vector.x, q.vector.y, q.vector.z, q.scalar], dtype=np.float32)

def np_to_magnum_quat(arr):
    
    return Quaternion(Vector3(arr[:3]), arr[3])

def magnum_vec3_to_np(v):
    return np.array([v[0], v[1], v[2]], dtype=np.float32)

def np_to_magnum_vec3(arr):
    return Vector3(arr)

# agent utils
def agent_state_to_serializable(agent_state):
    # Handle sensor_states
    sensor_states = {}
    if hasattr(agent_state, 'sensor_states') and agent_state.sensor_states:
        for sensor_name, sensor_pose in agent_state.sensor_states.items():
            sensor_states[sensor_name] = {
                "position": sensor_pose.position,  # already np.ndarray
                "rotation": np.quaternion(
                    sensor_pose.rotation.w,
                    sensor_pose.rotation.x,
                    sensor_pose.rotation.y,
                    sensor_pose.rotation.z
                ),  # quaternion.quaternion
            }
    
    return {
        "position": agent_state.position,  # already np.ndarray
        "rotation": np.quaternion(
            agent_state.rotation.w,
            agent_state.rotation.x,
            agent_state.rotation.y,
            agent_state.rotation.z
        ),  # quaternion.quaternion
        "sensor_states": sensor_states,
    }

def serializable_to_agent_state(agent_state_dict):
    s = AgentState()
    s.position = agent_state_dict["position"]
    s.rotation = agent_state_dict["rotation"]
    
    # Handle sensor_states
    if "sensor_states" in agent_state_dict and agent_state_dict["sensor_states"]:
        sensor_states = {}
        for sensor_name, sensor_pose_dict in agent_state_dict["sensor_states"].items():
            sensor_states[sensor_name] = SixDOFPose(
                position=sensor_pose_dict["position"],
                rotation=sensor_pose_dict["rotation"]
            )
        s.sensor_states = sensor_states
    
    return s


@ray.remote(num_cpus=0.01, num_gpus=0.001)
class HabitatIoUWorker:
    def __init__(self, dataset_name="ReplicaCAD", enable_semantic=True):
        if not ray.is_initialized():
            ray.init(_temp_dir=TEMP_DIR,)
        self.dataset_name = dataset_name
        self.sim = None

    def reset(self, agent_state_dict, obj_handle, obj_rotation_np, obj_translation_np):
        self.sim = create_hm3d_simulator("ReplicaCAD", "NONE", enable_semantic=True)

        # Agent
        agent_empty = self.sim.get_agent(0)
        agent_state_empty = serializable_to_agent_state(agent_state_dict)
        agent_empty.set_state(agent_state_empty, infer_sensor_states=False) #NOTE: infer_sensor_states=False is important for sync sensor states (eg.look_down, look_up)

        # Object
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        obj = rigid_obj_mgr.add_object_by_template_handle(obj_handle)
        obj.rotation = np_to_magnum_quat(obj_rotation_np)
        obj.translation = np_to_magnum_vec3(obj_translation_np)

        obs_empty = self.sim.get_sensor_observations()
        self.sim.close()
        return obs_empty
    
    def reset_class(self, agent_state_dict, obj_info_list):
        """
        Reset with multiple objects of the same class for class-detect task.
        逐个加载物体，分别获取每个物体的 GT bbox 和 mask。
        
        Args:
            agent_state_dict: Serializable agent state
            obj_info_list: List of dicts, each containing:
                - obj_handle: object template handle
                - obj_rotation_np: rotation as numpy array
                - obj_translation_np: translation as numpy array
                - instance_id: instance ID for this object
        
        Returns:
            obs_list: List of observations, each containing one object's GT
        """
        obs_list = []
        
        # 逐个加载物体并获取观察
        for obj_info in obj_info_list:
            # 为每个物体创建新的空场景
            self.sim = create_hm3d_simulator("ReplicaCAD", "NONE", enable_semantic=True)
            
            # Agent
            agent_empty = self.sim.get_agent(0)
            agent_state_empty = serializable_to_agent_state(agent_state_dict)
            agent_empty.set_state(agent_state_empty, infer_sensor_states=False)
            
            # 只添加当前这一个物体
            rigid_obj_mgr = self.sim.get_rigid_object_manager()
            obj = rigid_obj_mgr.add_object_by_template_handle(obj_info['obj_handle'])
            obj.rotation = np_to_magnum_quat(obj_info['obj_rotation_np'])
            obj.translation = np_to_magnum_vec3(obj_info['obj_translation_np'])
            
            # 获取观察
            obs_empty = self.sim.get_sensor_observations()
            
            # 保存观察和对应的实例ID
            obs_with_id = {
                'obs': obs_empty,
                'instance_id': obj_info['instance_id']
            }
            obs_list.append(obs_with_id)
            
            # 关闭场景
            self.sim.close()
        
        return obs_list
    
    def get_observation(self):
        observations = self.sim.get_sensor_observations()
        return observations

    def close(self):
        self.sim.close()



# ============================================================================
# HM3DIoUWorker - Ray worker for HM3D unoccluded mask generation
# ============================================================================

from .constants import SENSOR_RESOLUTION, SCENE_DATA_PATH, TEMP_DIR
from .hm3d_unoccluded_utils import HM3DUnoccludedMaskGenerator

from pycocotools import mask as mask_utils
import numpy as np
import math
import os
import json

@ray.remote(num_cpus=0.01, num_gpus=0.001)
class HM3DIoUWorker:
    """
    Ray worker for HM3D unoccluded mask generation.
    
    Unlike ReplicaCAD where objects are independent models, HM3D objects are
    part of the scene mesh. We use preprocessed meshes for stable rendering.
    """
    
    def __init__(self, data_path: str = SCENE_DATA_PATH):
        if not ray.is_initialized():
            ray.init(_temp_dir=TEMP_DIR)
        self.data_path = data_path
        self._generator_cache = {}  # scene_folder -> HM3DUnoccludedMaskGenerator
        self._metadata_cache = {}   # scene_folder -> metadata dict
        
    def _get_generator(self, scene_folder: str, semantic_objects: dict = None):
        """
        Get or create a cached generator for the scene.
        
        Args:
            scene_folder: Path to HM3D scene folder
            semantic_objects: Optional pre-extracted semantic objects dict
                              {semantic_id: (center, sizes, category)}
        """
        if scene_folder not in self._generator_cache:
            self._generator_cache[scene_folder] = HM3DUnoccludedMaskGenerator(
                scene_folder=scene_folder,
                data_path=self.data_path,
                resolution=SENSOR_RESOLUTION,
                sensor_height=1.0,
                hfov=90.0,
                semantic_objects=semantic_objects
            )
        return self._generator_cache[scene_folder]
    
    def _get_metadata(self, scene_folder: str) -> dict:
        """
        Get cached metadata for a scene.
        
        Args:
            scene_folder: Path to HM3D scene folder
            
        Returns:
            Metadata dict or empty dict if not found
        """
        if scene_folder not in self._metadata_cache:
            metadata_path = os.path.join(scene_folder, "object_mesh", "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self._metadata_cache[scene_folder] = json.load(f)
            else:
                self._metadata_cache[scene_folder] = {}
        return self._metadata_cache[scene_folder]
    
    def get_valid_semantic_ids(self, scene_folder: str) -> list:
        """
        Get list of valid (preprocessed) semantic IDs for a scene.
        
        Args:
            scene_folder: Path to HM3D scene folder
            
        Returns:
            List of valid semantic IDs
        """
        metadata = self._get_metadata(scene_folder)
        return [obj["semantic_id"] for obj in metadata.get("valid_objects", [])]
    
    def is_valid_semantic_id(self, scene_folder: str, semantic_id: int) -> bool:
        """
        Check if a semantic ID has a preprocessed mesh.
        
        Args:
            scene_folder: Path to HM3D scene folder
            semantic_id: Semantic ID to check
            
        Returns:
            True if the object has a preprocessed mesh
        """
        valid_ids = self.get_valid_semantic_ids(scene_folder)
        return semantic_id in valid_ids
    
    def get_valid_objects_by_category(self, scene_folder: str, category: str) -> list:
        """
        Get list of valid objects for a specific category.
        
        Args:
            scene_folder: Path to HM3D scene folder
            category: Category name (case-insensitive)
            
        Returns:
            List of object info dicts
        """
        metadata = self._get_metadata(scene_folder)
        category_lower = category.lower()
        return [
            obj for obj in metadata.get("valid_objects", [])
            if obj["category"].lower() == category_lower
        ]
    
    def reset(self, agent_state_dict, scene_folder: str, semantic_id: int, semantic_objects: dict = None):
        """
        Generate unoccluded mask for a single HM3D object.
        
        Args:
            agent_state_dict: Serializable agent state (position, rotation)
            scene_folder: Path to HM3D scene folder (e.g., "/data/.../00009-vLpv2VX547B")
            semantic_id: Semantic ID of the target object
            semantic_objects: Optional pre-extracted semantic objects dict
                              {semantic_id: (center, sizes, category)}
                              If provided, avoids redundant scene loading.
            
        Returns:
            dict containing:
                - rgb: RGB observation as numpy array (H, W, 4)
                - semantic: Semantic mask as numpy array (H, W) 
                - depth: Depth observation as numpy array (H, W)
                - bbox_gt: Bounding box (xmin, xmax, ymin, ymax)
                - mask_gt: RLE encoded mask dict
                - bbox_3d_gt: 3D bounding box dict
                - category: Object category name
        """
        generator = self._get_generator(scene_folder, semantic_objects)
        
        # Extract agent position and rotation from agent_state_dict
        # agent_state_dict has 'position' as np.ndarray and 'rotation' as quaternion
        position = agent_state_dict["position"]
        rotation_quat = agent_state_dict["rotation"]
        
        # Extract sensor_states for syncing look_down/look_up actions
        sensor_states = agent_state_dict.get("sensor_states", None)
        
        # Convert quaternion to rotation angle around Y axis
        # quaternion format: w, x, y, z for numpy-quaternion
        
        # For a rotation around Y axis: angle = 2 * atan2(y, w)
        rotation = 2 * math.atan2(rotation_quat.y, rotation_quat.w)
        
        # Get unoccluded mask
        result = generator.get_unoccluded_mask(
            semantic_id=semantic_id,
            agent_position=position,
            agent_rotation=rotation,
            sensor_states=sensor_states
        )
        
        # Convert PIL image to numpy for consistency with ReplicaCAD worker
        rgb_np = np.array(result['rgb'])
        # Add alpha channel to match habitat-sim RGBA format
        rgba_np = np.zeros((rgb_np.shape[0], rgb_np.shape[1], 4), dtype=np.uint8)
        rgba_np[:, :, :3] = rgb_np
        rgba_np[:, :, 3] = 255
        
        # Create semantic observation from mask
        mask_decoded = mask_utils.decode(result['mask_gt'])
        # 避免 uint8 乘法溢出：先提升 dtype 再乘
        semantic_np = mask_decoded.astype(np.uint32) * np.uint32(semantic_id)
 
        
        # Create observation dict similar to habitat-sim format
        obs = {
            "rgb": rgba_np,
            "semantic": semantic_np,
            "depth": None,  # Depth not easily available from extracted mesh
        }
        
        return {
            "obs": obs,
            "bbox_gt": result['bbox_gt'],
            "mask_gt": result['mask_gt'],
            "bbox_3d_gt": result['bbox_3d_gt'],
            "category": result['category']
        }
    
    def reset_class(self, agent_state_dict, scene_folder: str, semantic_ids: list):
        """
        Generate unoccluded masks for multiple HM3D objects (same class).
        
        Args:
            agent_state_dict: Serializable agent state
            scene_folder: Path to HM3D scene folder
            semantic_ids: List of semantic IDs to process
            
        Returns:
            List of result dicts, one per semantic_id
        """
        results = []
        for semantic_id in semantic_ids:
            try:
                result = self.reset(agent_state_dict, scene_folder, semantic_id)
                result['semantic_id'] = semantic_id
                results.append(result)
            except Exception as e:
                print(f"HM3DIoUWorker: Error processing semantic_id {semantic_id}: {e}")
                continue
        return results
    
    def get_object_aabb(self, scene_folder: str, semantic_id: int):
        """
        Get AABB for an object in HM3D scene.
        
        Returns:
            Tuple of (center, sizes, category) or None if not found
        """
        try:
            generator = self._get_generator(scene_folder)
            return generator.get_object_aabb(semantic_id)
        except Exception as e:
            print(f"HM3DIoUWorker: Error getting AABB for semantic_id {semantic_id}: {e}")
            return None
    
    def cleanup(self, scene_folder: str = None):
        """Clean up temporary files and cached generators."""
        if scene_folder is not None and scene_folder in self._generator_cache:
            self._generator_cache[scene_folder].cleanup()
            del self._generator_cache[scene_folder]
        elif scene_folder is None:
            for gen in self._generator_cache.values():
                gen.cleanup()
            self._generator_cache.clear()