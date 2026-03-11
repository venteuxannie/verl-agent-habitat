"""
Evaluation Dataset Collection Script (Ray Version)

This script collects evaluation data from HM3D scenes in parallel using Ray.
Each worker processes one scene independently with proper GPU access.

Usage:
    python eval_dataset.py [--workers N] [--scenes N] [--instances N]
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"

import sys
import argparse
from typing import Dict, List
from PIL import Image
import json
import numpy as np
import time
import threading

import ray

from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import draw_bbox_with_text
from agent_system.environments.env_package.habitat_sim.dataset.collect_sft_dataset_rule import annotate_image
from agent_system.environments.env_package.habitat_sim.utils.constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE
from agent_system.environments.env_package.habitat_sim.utils.constants import SCENE_DATA_PATH, EVAL_DATA_PATH, HM3D_TRAINING_SCENES, REPLICACAD_TRAINING_SCENES, TEMP_DIR


ANNOTATION_OPTIONS = {
    "draw_text": True,
    "draw_bbox": True,
    "draw_mask": True,
    "draw_3dbox": True,
    "draw_all_annotations": True,
    "bbox_color": (0, 255, 0),
    "mask_color": (255, 0, 255),
    "box3d_color": (0, 0, 255),
}


class NumpyEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def get_scene_path(subfolders, dataset_name, eval_id=0):
    """Constructs the full path to a habitat scene file."""
    data_path = SCENE_DATA_PATH
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


@ray.remote(num_cpus=0.01, num_gpus=0.001)
class SceneCollector:
    """
    Ray Actor for collecting data from a single scene.
    Each actor has its own environment and GPU access.
    """
    
    def __init__(self, dataset_name: str, scenes_size: int, max_scene_instance: int, max_step_length: int, seed: int):
        self.dataset_name = dataset_name
        self.scenes_size = scenes_size
        self.max_scene_instance = max_scene_instance
        self.max_step_length = max_step_length
        self.seed = seed
        self.env = None
    
    def _ensure_env(self):
        """Create environment if not exists."""
        if self.env is None:
            self.env = CreateHabitatEnv(
                self.seed, 
                self.dataset_name, 
                self.scenes_size, 
                self.max_scene_instance, 
                self.max_step_length
            )
    
    def collect_scene(self, scene_idx: int, scene_subfolder: List[str], image_save_path: str) -> Dict:
        """
        Collect data for a single scene.
        
        Returns:
            Dict with scene results and task infos
        """
        self._ensure_env()
        
        scene_name = scene_subfolder[scene_idx]
        special_scene_id = get_scene_path(scene_subfolder, self.dataset_name, eval_id=scene_idx)
        
        task_infos = []
        collected = 0
        
        for j in range(self.max_scene_instance):
            try:
                obs, info = self.env.reset(
                    seed=self.seed,
                    is_unique=True,
                    sync_info=None,
                    special_scene_id=special_scene_id,
                    task_type="grounding"
                )
                info["task_type"] = "any"
                
                # Calculate global task_id
                task_id = scene_idx * self.max_scene_instance + j
                info['task_id'] = task_id
                
                task_infos.append(info)
                
                # Annotate image
                gt_data = info.get("gt", {}) or {}
                bbox_2d_for_annotation = gt_data.get("bbox_2d_gt")
                mask_rle_for_annotation = gt_data.get("mask_gt")
                bbox_3d_for_annotation = gt_data.get("bbox_3d_gt")
                
                image_width, image_height = obs.size
                annotated_obs = annotate_image(
                    image=obs,
                    task_type=info.get("task_type"),
                    target_category=info.get("target_category"),
                    task_prompt=info.get("task_prompt"),
                    pred=info.get("pred"),
                    conf_score=info.get("conf_score") or 0.0,
                    step_id=info.get("step_id"),
                    image_size=(image_width, image_height),
                    hfov=90.0,
                    bbox_2d=bbox_2d_for_annotation,
                    mask_rle=mask_rle_for_annotation,
                    bbox_3d=bbox_3d_for_annotation,
                    draw_text=ANNOTATION_OPTIONS.get("draw_text", True),
                    draw_bbox=ANNOTATION_OPTIONS.get("draw_bbox", True),
                    draw_mask=ANNOTATION_OPTIONS.get("draw_mask", True),
                    draw_3dbox=ANNOTATION_OPTIONS.get("draw_3dbox", True),
                    draw_all_annotations=ANNOTATION_OPTIONS.get("draw_all_annotations", True),
                    bbox_color=ANNOTATION_OPTIONS.get("bbox_color", (0, 255, 0)),
                    mask_color=ANNOTATION_OPTIONS.get("mask_color", (255, 0, 255)),
                    box3d_color=ANNOTATION_OPTIONS.get("box3d_color", (0, 0, 255)),
                )
                
                # Save image
                annotated_obs.save(os.path.join(image_save_path, f"{task_id + 1}.png"))
                collected += 1
                
            except Exception as e:
                print(f"Error in scene {scene_name}, instance {j}: {e}")
                continue
        
        return {
            "scene_name": scene_name,
            "scene_idx": scene_idx,
            "status": "success",
            "task_infos": task_infos,
            "count": collected
        }
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None


def collect_eval_data(
    dataset_name: str = "HM3D",
    eval_data_name: str = "hm3d_10-any-500-seen",
    scenes_size: int = 10,
    max_scene_instance: int = 50,
    max_step_length: int = 10,
    seed: int = 0,
    num_workers: int = None
):
    """
    Collect evaluation data using Ray actors.
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(_temp_dir=TEMP_DIR, ignore_reinit_error=True)
    
    # Setup paths
    dir_path = EVAL_DATA_PATH
    eval_data_path = os.path.join(dir_path, eval_data_name)
    os.makedirs(eval_data_path, exist_ok=True)
    image_save_path = os.path.join(eval_data_path, "images")
    os.makedirs(image_save_path, exist_ok=True)
    
    # Get scene list
    if dataset_name == "HM3D":
        scene_subfolder = HM3D_TRAINING_SCENES[:scenes_size]
    elif dataset_name == "ReplicaCAD":
        scene_subfolder = REPLICACAD_TRAINING_SCENES[:scenes_size]
    else:
        raise ValueError("Only HM3D and ReplicaCAD are supported for now")
    
    if num_workers is None:
        num_workers = min(scenes_size, 10)
    
    print(f"Collecting data for {scenes_size} scenes with {num_workers} Ray workers...")
    print(f"Output path: {eval_data_path}")
    print()
    
    # Create Ray actors
    collectors = [
        SceneCollector.remote(
            dataset_name, scenes_size, max_scene_instance, max_step_length, seed
        )
        for _ in range(num_workers)
    ]
    
    # Submit tasks round-robin to workers
    pending_tasks = {}  # task_ref -> (scene_idx, scene_name)
    scene_queue = list(range(scenes_size))
    results = []
    
    # Initial task submission
    for i, scene_idx in enumerate(scene_queue[:num_workers]):
        collector = collectors[i % num_workers]
        task_ref = collector.collect_scene.remote(scene_idx, scene_subfolder, image_save_path)
        pending_tasks[task_ref] = (scene_idx, scene_subfolder[scene_idx])
    
    remaining_scenes = scene_queue[num_workers:]
    
    # Progress tracking
    completed = 0
    
    # Process results as they complete
    while pending_tasks:
        # Wait for any task to complete
        done_refs, _ = ray.wait(list(pending_tasks.keys()), num_returns=1)
        
        for done_ref in done_refs:
            scene_idx, scene_name = pending_tasks.pop(done_ref)
            
            try:
                result = ray.get(done_ref)
                results.append(result)
                completed += 1
                
                if result["status"] == "success":
                    print(f"[{completed}/{scenes_size}] {scene_name}: {result['count']} tasks collected ✓")
                else:
                    print(f"[{completed}/{scenes_size}] {scene_name}: ERROR")
                    
            except Exception as e:
                completed += 1
                print(f"[{completed}/{scenes_size}] {scene_name}: FAILED - {e}")
                results.append({
                    "scene_name": scene_name,
                    "scene_idx": scene_idx,
                    "status": "error",
                    "error": str(e),
                    "task_infos": [],
                    "count": 0
                })
            
            # Submit next task if available
            if remaining_scenes:
                next_scene_idx = remaining_scenes.pop(0)
                # Find a free collector (round-robin based on completed count)
                collector = collectors[completed % num_workers]
                task_ref = collector.collect_scene.remote(
                    next_scene_idx, scene_subfolder, image_save_path
                )
                pending_tasks[task_ref] = (next_scene_idx, scene_subfolder[next_scene_idx])
    
    # Clean up actors
    for collector in collectors:
        ray.get(collector.close.remote())
    
    # Collect all task infos and sort by task_id
    all_task_infos = []
    for result in results:
        if result["status"] == "success":
            all_task_infos.extend(result["task_infos"])
    
    # Sort by task_id
    all_task_infos.sort(key=lambda x: x.get("task_id", 0))
    
    # Print summary
    print()
    print(f"{'='*60}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    
    total_tasks = 0
    for result in sorted(results, key=lambda x: x["scene_idx"]):
        if result["status"] == "success":
            print(f"  {result['scene_name']}: {result['count']} tasks ✓")
            total_tasks += result["count"]
        else:
            print(f"  {result['scene_name']}: ERROR - {result.get('error', 'unknown')}")
    
    print(f"\nTotal: {total_tasks} tasks collected")
    
    # Save all task infos to JSON
    output_filename = os.path.join(eval_data_path, 'task_infos.json')
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_task_infos, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
    print(f"Task infos saved to: {output_filename}")
    
    return all_task_infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect evaluation dataset using Ray")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HM3D",
        help="Dataset name (HM3D or ReplicaCAD)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="hm3d_10-any-500-seen",
        help="Name for the evaluation dataset"
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=10,
        help="Number of scenes to process"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=50,
        help="Number of instances per scene"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of Ray workers (default: number of scenes, max 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    collect_eval_data(
        dataset_name=args.dataset,
        eval_data_name=args.name,
        scenes_size=args.scenes,
        max_scene_instance=args.instances,
        seed=args.seed,
        num_workers=args.workers
    )

