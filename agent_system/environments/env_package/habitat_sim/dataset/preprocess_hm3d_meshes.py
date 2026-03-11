"""
HM3D Object Mesh Preprocessing Script

This script pre-extracts object meshes from HM3D scenes and saves them for later use
by HM3DIoUWorker. This avoids runtime mesh extraction which can be slow and error-prone.

Features:
- Multiprocessing for parallel scene processing
- Real-time per-scene progress display

Usage:
    python preprocess_hm3d_meshes.py [--scenes SCENE1 SCENE2 ...] [--force] [--workers N]

Output structure:
    {scene_folder}/object_mesh/
        ├── metadata.json
        ├── {semantic_id}.obj
        └── ...

Author: TCT
Date: 2026-01-09
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
import warnings
import threading
import time
import io
import contextlib

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress habitat-sim logs
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"

import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec

from agent_system.environments.env_package.habitat_sim.utils.constants import (
    HM3D_TRAINING_SCENES,
    SCENE_DATA_PATH,
    SENSOR_RESOLUTION,
    hm3d_test,
)
from agent_system.environments.env_package.habitat_sim.utils.hm3d_unoccluded_utils import (
    HM3DSemanticParser,
    HM3DMeshExtractor,
)

# Target categories to preprocess
TARGET_CATEGORIES = list(hm3d_test.keys())


def get_scene_folder(scene_name: str, data_path: str = SCENE_DATA_PATH) -> str:
    """Get full path to scene folder."""
    return os.path.join(data_path, "hm3d/train", scene_name)


def load_scene_semantic_objects(
    scene_folder: str,
    data_path: str = SCENE_DATA_PATH,
    resolution: Tuple[int, int] = SENSOR_RESOLUTION
) -> Dict[int, Tuple[np.ndarray, np.ndarray, str]]:
    """Load semantic objects from a HM3D scene."""
    scene_name = os.path.basename(scene_folder)
    scene_id = scene_name.split('-')[1] if '-' in scene_name else scene_name
    basis_glb_path = os.path.join(scene_folder, f"{scene_id}.basis.glb")
    
    if not os.path.exists(basis_glb_path):
        raise FileNotFoundError(f"Scene file not found: {basis_glb_path}")
    
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.join(
        data_path, "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    )
    sim_cfg.scene_id = basis_glb_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = list(resolution)
    rgb_spec.position = [0.0, 1.0, 0.0]
    agent_cfg.sensor_specifications = [rgb_spec]
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
    }
    
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    
    semantic_objects = {}
    try:
        for obj in sim.semantic_scene.objects:
            if obj and obj.category and obj.aabb.sizes[0] > 0:
                semantic_objects[obj.semantic_id] = (
                    np.array(obj.aabb.center),
                    np.array(obj.aabb.sizes),
                    obj.category.name()
                )
    finally:
        sim.close()
    
    return semantic_objects


def filter_target_objects(
    semantic_objects: Dict[int, Tuple[np.ndarray, np.ndarray, str]],
    parser: HM3DSemanticParser,
    target_categories: List[str]
) -> Dict[int, Tuple[np.ndarray, np.ndarray, str]]:
    """Filter semantic objects to only include target categories."""
    target_categories_lower = [cat.lower() for cat in target_categories]
    return {
        sid: (center, sizes, cat) 
        for sid, (center, sizes, cat) in semantic_objects.items()
        if cat.lower() in target_categories_lower
    }


def preprocess_scene_with_progress(args):
    """Process a single scene with progress reporting via shared dict."""
    scene_name, data_path, force, progress_dict = args
    
    scene_folder = get_scene_folder(scene_name, data_path)
    output_dir = os.path.join(scene_folder, "object_mesh")
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    # Initialize progress
    progress_dict[scene_name] = {"current": 0, "total": 0, "status": "loading"}
    
    # Check if already preprocessed
    if os.path.exists(metadata_path) and not force:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        progress_dict[scene_name] = {
            "current": len(metadata["valid_objects"]) + len(metadata["failed_objects"]),
            "total": metadata["total_target_objects"],
            "status": "skipped"
        }
        return {"scene_name": scene_name, "status": "skipped", "metadata": metadata}
    
    os.makedirs(output_dir, exist_ok=True)
    
    scene_id = scene_name.split('-')[1] if '-' in scene_name else scene_name
    semantic_txt_path = os.path.join(scene_folder, f"{scene_id}.semantic.txt")
    semantic_glb_path = os.path.join(scene_folder, f"{scene_id}.semantic.glb")
    
    if not os.path.exists(semantic_txt_path) or not os.path.exists(semantic_glb_path):
        progress_dict[scene_name]["status"] = "error"
        return {"scene_name": scene_name, "status": "error", "error": "Files not found"}
    
    try:
        parser = HM3DSemanticParser(semantic_txt_path)
        extractor = HM3DMeshExtractor(semantic_glb_path, parser)
        semantic_objects = load_scene_semantic_objects(scene_folder, data_path)
        target_objects = filter_target_objects(semantic_objects, parser, TARGET_CATEGORIES)
        
        # Update total
        progress_dict[scene_name] = {"current": 0, "total": len(target_objects), "status": "processing"}
        
        valid_objects = []
        failed_objects = []
        
        for idx, (semantic_id, (center, sizes, category)) in enumerate(target_objects.items()):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    mesh = extractor.extract_object_mesh_by_color(
                        semantic_id=semantic_id,
                        color_tolerance=2,
                        aabb_filter=(center, sizes),
                        aabb_padding=0.3,
                        min_vertices=50
                    )
                
                mesh_center = mesh.bounds.mean(axis=0)
                mesh.vertices -= mesh_center
                
                mesh_filename = f"{semantic_id}.obj"
                with contextlib.redirect_stdout(io.StringIO()):
                    extractor.save_object_mesh(mesh, os.path.join(output_dir, mesh_filename))
                
                valid_objects.append({
                    "semantic_id": int(semantic_id),
                    "category": category,
                    "aabb_center": center.tolist(),
                    "aabb_sizes": sizes.tolist(),
                    "mesh_offset": mesh_center.tolist(),
                    "mesh_file": mesh_filename,
                    "num_vertices": len(mesh.vertices),
                    "num_faces": len(mesh.faces)
                })
            except Exception as e:
                failed_objects.append({
                    "semantic_id": int(semantic_id),
                    "category": category,
                    "error": str(e)
                })
            
            # Update progress
            progress_dict[scene_name]["current"] = idx + 1
        
        metadata = {
            "scene_id": scene_name,
            "scene_folder": scene_folder,
            "preprocessed_at": datetime.now().isoformat(),
            "target_categories": TARGET_CATEGORIES,
            "total_target_objects": len(target_objects),
            "valid_objects": valid_objects,
            "failed_objects": failed_objects,
            "success_rate": len(valid_objects) / len(target_objects) if target_objects else 0
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        progress_dict[scene_name]["status"] = "done"
        return {"scene_name": scene_name, "status": "success", "metadata": metadata}
        
    except Exception as e:
        progress_dict[scene_name]["status"] = "error"
        return {"scene_name": scene_name, "status": "error", "error": str(e)}


def display_progress(scenes: List[str], progress_dict: dict, num_workers: int, stop_event: threading.Event):
    """Thread function to display real-time progress for all scenes."""
    while not stop_event.is_set():
        lines = []
        for scene_name in scenes:
            if scene_name in progress_dict:
                p = dict(progress_dict[scene_name])  # Copy to avoid race condition
                current = p.get("current", 0)
                total = p.get("total", 0)
                status = p.get("status", "waiting")
                
                if status == "done":
                    bar = "█" * 20
                    lines.append(f"  {scene_name}: [{bar}] Done ✓")
                elif status == "skipped":
                    lines.append(f"  {scene_name}: [----Skipped----] Already preprocessed")
                elif status == "error":
                    lines.append(f"  {scene_name}: [-----Error-----] ✗")
                elif status == "processing" and total > 0:
                    pct = current / total
                    filled = int(pct * 20)
                    bar = "█" * filled + "░" * (20 - filled)
                    lines.append(f"  {scene_name}: [{bar}] {current}/{total}")
                elif status == "loading":
                    lines.append(f"  {scene_name}: [    Loading    ]")
                else:
                    lines.append(f"  {scene_name}: [    Waiting    ]")
            else:
                lines.append(f"  {scene_name}: [    Waiting    ]")
        
        # Clear screen and print
        print("\033[H\033[J", end="")  # Clear screen
        print(f"Preprocessing {len(scenes)} HM3D scenes with {num_workers} workers...\n")
        print("\n".join(lines))
        sys.stdout.flush()
        time.sleep(0.3)


def preprocess_all_scenes(
    scenes: Optional[List[str]] = None,
    data_path: str = SCENE_DATA_PATH,
    force: bool = False,
    num_workers: int = None
) -> Dict:
    """
    Preprocess all HM3D training scenes using multiprocessing with real-time progress.
    
    Args:
        scenes: List of scene names to process. If None, use HM3D_TRAINING_SCENES
        data_path: Path to habitat data directory
        force: If True, overwrite existing preprocessed data
        num_workers: Number of parallel workers. If None, use number of scenes
    
    Returns:
        Summary dict with results for all scenes
    """
    if scenes is None:
        scenes = HM3D_TRAINING_SCENES
    
    if num_workers is None:
        num_workers = min(len(scenes), cpu_count())
    
    # Use Manager for shared progress dict
    manager = Manager()
    progress_dict = manager.dict()
    
    args_list = [(scene_name, data_path, force, progress_dict) for scene_name in scenes]
    
    # Start display thread
    stop_display = threading.Event()
    display_thread = threading.Thread(
        target=display_progress, 
        args=(scenes, progress_dict, num_workers, stop_display)
    )
    display_thread.start()
    
    # Process scenes in parallel
    results = {}
    try:
        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(preprocess_scene_with_progress, args_list):
                results[result["scene_name"]] = result
    finally:
        # Stop display thread
        stop_display.set()
        display_thread.join()
    
    # Print final summary
    print("\033[H\033[J", end="")  # Clear screen
    print(f"{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    
    total_valid = 0
    total_failed = 0
    for scene_name in scenes:
        result = results.get(scene_name, {})
        if result.get("status") == "success":
            m = result["metadata"]
            v, f = len(m["valid_objects"]), len(m["failed_objects"])
            total_valid += v
            total_failed += f
            print(f"  {scene_name}: {v} valid, {f} failed ✓")
        elif result.get("status") == "skipped":
            m = result["metadata"]
            v, f = len(m["valid_objects"]), len(m["failed_objects"])
            total_valid += v
            total_failed += f
            print(f"  {scene_name}: {v} valid, {f} failed (skipped)")
        else:
            print(f"  {scene_name}: ERROR - {result.get('error', 'unknown')}")
    
    print(f"\nTotal: {total_valid} valid, {total_failed} failed")
    return results


def load_preprocessed_metadata(scene_folder: str) -> Optional[Dict]:
    """Load preprocessed metadata for a scene."""
    metadata_path = os.path.join(scene_folder, "object_mesh", "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def get_valid_semantic_ids(scene_folder: str) -> List[int]:
    """Get list of valid (preprocessed) semantic IDs for a scene."""
    metadata = load_preprocessed_metadata(scene_folder)
    if metadata is None:
        return []
    return [obj["semantic_id"] for obj in metadata.get("valid_objects", [])]


def get_valid_objects_by_category(scene_folder: str, category: str) -> List[Dict]:
    """Get list of valid objects for a specific category."""
    metadata = load_preprocessed_metadata(scene_folder)
    if metadata is None:
        return []
    
    category_lower = category.lower()
    return [
        obj for obj in metadata.get("valid_objects", [])
        if obj["category"].lower() == category_lower
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HM3D object meshes")
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Specific scenes to process. If not provided, processes all HM3D_TRAINING_SCENES"
    )
    parser.add_argument(
        "--data-path",
        default=SCENE_DATA_PATH,
        help=f"Path to habitat data directory (default: {SCENE_DATA_PATH})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-preprocessing even if metadata already exists"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of scenes or CPU count)"
    )
    
    args = parser.parse_args()
    
    preprocess_all_scenes(
        scenes=args.scenes,
        data_path=args.data_path,
        force=args.force,
        num_workers=args.workers
    )
