"""
HM3D Unoccluded Mask Generator

This module provides functionality to extract unoccluded masks, bboxes, and 3D boxes
for objects in HM3D scanned scenes. Unlike ReplicaCAD where objects are independent
models, HM3D objects are part of the scene mesh. We achieve "unoccluded" rendering by:
1. Extracting the target object's mesh from semantic.glb based on vertex colors
2. Saving the extracted mesh as a temporary GLB file
3. Loading it as a standalone scene in habitat-sim
4. Rendering from the specified viewpoint to get unoccluded mask

Author: TCT
Date: 2026-01-04
"""

import os
import numpy as np
import tempfile
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pycocotools import mask as mask_utils
from scipy.spatial import ConvexHull

# Suppress habitat-sim logs
os.environ["HABITAT_SIM_LOG"] = "quiet"

import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec, SixDOFPose
import magnum as mn

from .constants import SCENE_DATA_PATH, SENSOR_RESOLUTION, MESH_TEMP_DIR, TEMP_DIR

# ============================================================================
# HM3DSemanticParser - Parse semantic.txt file
# ============================================================================

@dataclass
class SemanticInfo:
    """Data class to store semantic information for an object"""
    semantic_id: int
    color_hex: str
    color_rgb: Tuple[int, int, int]
    category: str
    region_id: int


class HM3DSemanticParser:
    """
    Parser for HM3D semantic.txt files.
    
    The semantic.txt format is:
    Line 1: "HM3D Semantic Annotations"
    Line 2+: semantic_id,color_hex,"category",region_id
    
    Example:
    69,39A436,"chair",0
    """
    
    def __init__(self, semantic_txt_path: str):
        """
        Initialize the parser with path to semantic.txt file.
        
        Args:
            semantic_txt_path: Path to the semantic.txt file
        """
        self.semantic_txt_path = semantic_txt_path
        self.semantic_info: Dict[int, SemanticInfo] = {}
        self.category_to_ids: Dict[str, List[int]] = {}
        
        self._parse_file()
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.strip()
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)
        else:
            raise ValueError(f"Invalid hex color: {hex_color}")
    
    def _parse_file(self):
        """Parse the semantic.txt file."""
        with open(self.semantic_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("HM3D"):
                continue
            
            # Parse: semantic_id,color_hex,"category",region_id
            # Handle quoted category names
            parts = line.split(',')
            if len(parts) < 4:
                continue
            
            try:
                semantic_id = int(parts[0])
                color_hex = parts[1].strip()
                
                # Find category (may contain quotes)
                # Category is between the second comma and the last comma
                category_start = line.find(',', line.find(',') + 1) + 1
                category_end = line.rfind(',')
                category = line[category_start:category_end].strip().strip('"')
                
                region_id = int(parts[-1])
                
                color_rgb = self._hex_to_rgb(color_hex)
                
                info = SemanticInfo(
                    semantic_id=semantic_id,
                    color_hex=color_hex,
                    color_rgb=color_rgb,
                    category=category,
                    region_id=region_id
                )
                
                self.semantic_info[semantic_id] = info
                
                # Build category to IDs mapping
                category_lower = category.lower()
                if category_lower not in self.category_to_ids:
                    self.category_to_ids[category_lower] = []
                self.category_to_ids[category_lower].append(semantic_id)
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line: {line}, error: {e}")
                continue
    
    def get_color_by_id(self, semantic_id: int) -> Tuple[int, int, int]:
        """
        Get RGB color for a semantic ID.
        
        Args:
            semantic_id: The semantic ID to look up
            
        Returns:
            RGB tuple (r, g, b) where each value is 0-255
        """
        if semantic_id in self.semantic_info:
            return self.semantic_info[semantic_id].color_rgb
        raise KeyError(f"Semantic ID {semantic_id} not found")
    
    def get_color_normalized(self, semantic_id: int) -> Tuple[float, float, float]:
        """
        Get normalized RGB color (0-1 range) for a semantic ID.
        
        Args:
            semantic_id: The semantic ID to look up
            
        Returns:
            RGB tuple (r, g, b) where each value is 0.0-1.0
        """
        r, g, b = self.get_color_by_id(semantic_id)
        return (r / 255.0, g / 255.0, b / 255.0)
    
    def get_category_by_id(self, semantic_id: int) -> str:
        """
        Get category name for a semantic ID.
        
        Args:
            semantic_id: The semantic ID to look up
            
        Returns:
            Category name string
        """
        if semantic_id in self.semantic_info:
            return self.semantic_info[semantic_id].category
        raise KeyError(f"Semantic ID {semantic_id} not found")
    
    def get_ids_by_category(self, category: str) -> List[int]:
        """
        Get all semantic IDs for a given category.
        
        Args:
            category: Category name (case-insensitive)
            
        Returns:
            List of semantic IDs belonging to that category
        """
        category_lower = category.lower()
        return self.category_to_ids.get(category_lower, [])
    
    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories."""
        return list(self.category_to_ids.keys())
    
    def get_all_ids(self) -> List[int]:
        """Get list of all semantic IDs."""
        return list(self.semantic_info.keys())
    
    def get_info(self, semantic_id: int) -> SemanticInfo:
        """Get full semantic info for an ID."""
        if semantic_id in self.semantic_info:
            return self.semantic_info[semantic_id]
        raise KeyError(f"Semantic ID {semantic_id} not found")


# ============================================================================
# HM3DMeshExtractor - Extract object mesh from semantic.glb
# ============================================================================

class HM3DMeshExtractor:
    """
    Extract object meshes from HM3D semantic.glb files based on vertex colors.
    
    The semantic.glb file contains the entire scene mesh with vertex colors
    corresponding to semantic annotations in semantic.txt. We extract objects
    by matching vertex colors to the target semantic ID.
    
    Coordinate systems:
    - GLB files use Y-up coordinate system
    - Habitat-sim applies a 90-degree rotation when loading GLB as scene
    - Transform: hab_x = glb_x, hab_y = glb_z, hab_z = -glb_y
    """
    
    def __init__(self, semantic_glb_path: str, semantic_parser: HM3DSemanticParser):
        """
        Initialize the mesh extractor.
        
        Args:
            semantic_glb_path: Path to the semantic.glb file
            semantic_parser: Parsed semantic information
        """
        self.semantic_glb_path = semantic_glb_path
        self.semantic_parser = semantic_parser
        
        # Lazy loading - mesh data
        self._meshes = None
        self._vertices = None       # Combined vertices (N, 3) in GLB coords
        self._vertex_colors = None  # Combined vertex colors (N, 3) RGB 0-255
        self._faces = None          # Combined faces (M, 3)
    
    def _get_vertex_colors(self, mesh) -> np.ndarray:
        """
        Extract vertex colors from a trimesh object.
        
        Args:
            mesh: trimesh.Trimesh object
            
        Returns:
            Vertex colors array (N, 3) with RGB values 0-255
        """
        n_vertices = len(mesh.vertices)
        
        # Method 1: Direct vertex_colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = np.array(mesh.visual.vertex_colors)[:, :3]
            if len(colors) == n_vertices:
                return colors
        
        # Method 2: Convert via to_color()
        try:
            color_visual = mesh.visual.to_color()
            if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                colors = np.array(color_visual.vertex_colors)[:, :3]
                if len(colors) == n_vertices:
                    return colors
        except Exception:
            pass
        
        # Method 3: Use material main_color
        try:
            if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'main_color'):
                main_color = np.array(mesh.visual.material.main_color)[:3]
                return np.tile(main_color, (n_vertices, 1))
        except Exception:
            pass
        
        # Fallback: return zeros (black)
        return np.zeros((n_vertices, 3), dtype=np.uint8)
    
    def _load_meshes(self):
        """Load all meshes from the GLB file and extract vertex colors."""
        if self._meshes is not None:
            return
        
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is required for mesh extraction. "
                "Install it with: pip install trimesh"
            )
        
        # Load the GLB file as a scene
        scene = trimesh.load(
            self.semantic_glb_path,
            force='scene'
        )
        
        # Dump all meshes (with transforms applied)
        self._meshes = scene.dump(concatenate=False)
        
        # Collect vertices, colors, and faces from all meshes
        all_vertices = []
        all_colors = []
        all_faces = []
        vertex_offset = 0
        
        for mesh in self._meshes:
            if not isinstance(mesh, trimesh.Trimesh):
                continue
            
            vertices = np.array(mesh.vertices)
            colors = self._get_vertex_colors(mesh)
            faces = np.array(mesh.faces) + vertex_offset
            
            if len(vertices) > 0:
                all_vertices.append(vertices)
                all_colors.append(colors)
                all_faces.append(faces)
                vertex_offset += len(vertices)
        
        if all_vertices:
            self._vertices = np.vstack(all_vertices)
            self._vertex_colors = np.vstack(all_colors)
            self._faces = np.vstack(all_faces)
        else:
            self._vertices = np.array([]).reshape(0, 3)
            self._vertex_colors = np.array([]).reshape(0, 3)
            self._faces = np.array([]).reshape(0, 3)
        
        # Count unique colors for debugging
        unique_colors = np.unique(self._vertex_colors, axis=0)
        print(f"Loaded {len(self._meshes)} mesh chunks from semantic.glb")
        print(f"  Total vertices: {len(self._vertices)}, Unique colors: {len(unique_colors)}")
    
    def _find_closest_mesh_color(self, target_color: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find the closest color in the mesh to the target color.
        
        Args:
            target_color: Target RGB color (0-255)
            
        Returns:
            Tuple of (closest_color, min_difference)
        """
        self._load_meshes()
        
        # Get unique colors in the mesh
        unique_colors = np.unique(self._vertex_colors, axis=0)
        
        # Compute max channel difference for each unique color
        color_diff = np.abs(unique_colors.astype(np.float32) - target_color.astype(np.float32))
        max_diff = np.max(color_diff, axis=1)
        
        # Find the closest color
        min_idx = np.argmin(max_diff)
        closest_color = unique_colors[min_idx]
        min_diff = max_diff[min_idx]
        
        return closest_color, min_diff
    
    def _find_best_color_tolerance(
        self, 
        target_color: np.ndarray, 
        min_vertices: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Find the best tolerance to match at least min_vertices.
        
        Args:
            target_color: Target RGB color (0-255)
            min_vertices: Minimum number of vertices to match
            
        Returns:
            Tuple of (best_color, best_tolerance)
        """
        self._load_meshes()
        
        # First, find the closest color in the mesh
        closest_color, min_diff = self._find_closest_mesh_color(target_color)
        
        # If the closest color is too different, use target color with adaptive tolerance
        if min_diff > 15:
            use_color = target_color
        else:
            use_color = closest_color
        
        # Compute differences from chosen color
        color_diff = np.abs(self._vertex_colors.astype(np.float32) - use_color.astype(np.float32))
        max_diff = np.max(color_diff, axis=1)
        
        # Try increasing tolerances until we have enough vertices
        for tolerance in [2, 3, 5, 8, 10, 12, 15, 20, 25]:
            count = np.sum(max_diff <= tolerance)
            if count >= min_vertices:
                return use_color, tolerance
        
        # If still not enough, return with max tolerance
        return use_color, 25
    
    def _vertices_in_aabb_glb(
        self, 
        aabb_center: np.ndarray, 
        aabb_sizes: np.ndarray,
        padding: float = 0.1
    ) -> np.ndarray:
        """
        Get boolean mask of vertices within AABB (in GLB coordinates).
        
        Args:
            aabb_center: AABB center in Habitat coordinates
            aabb_sizes: AABB sizes in Habitat coordinates  
            padding: Extra padding around AABB
            
        Returns:
            Boolean mask array (N,) for all vertices
        """
        # Convert AABB from Habitat to GLB coordinates
        glb_center = self.habitat_to_glb_coords(aabb_center)
        glb_sizes = np.array([aabb_sizes[0], aabb_sizes[2], aabb_sizes[1]])
        
        glb_min = glb_center - glb_sizes / 2 - padding
        glb_max = glb_center + glb_sizes / 2 + padding
        
        in_box = (
            (self._vertices[:, 0] >= glb_min[0]) & (self._vertices[:, 0] <= glb_max[0]) &
            (self._vertices[:, 1] >= glb_min[1]) & (self._vertices[:, 1] <= glb_max[1]) &
            (self._vertices[:, 2] >= glb_min[2]) & (self._vertices[:, 2] <= glb_max[2])
        )
        
        return in_box
    
    @staticmethod
    def habitat_to_glb_coords(point: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from Habitat to GLB coordinate system.
        
        Habitat applies a 90-degree rotation when loading GLB:
        - hab_x = glb_x
        - hab_y = glb_z  
        - hab_z = -glb_y
        
        Inverse transform:
        - glb_x = hab_x
        - glb_y = -hab_z
        - glb_z = hab_y
        """
        return np.array([point[0], -point[2], point[1]])
    
    @staticmethod
    def glb_to_habitat_coords(point: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from GLB to Habitat coordinate system.
        """
        return np.array([point[0], point[2], -point[1]])
    
    @staticmethod
    def glb_to_habitat_vertices(vertices: np.ndarray) -> np.ndarray:
        """
        Transform vertices array from GLB to Habitat coordinate system.
        
        Args:
            vertices: (N, 3) vertices in GLB coords
            
        Returns:
            (N, 3) vertices in Habitat coords
        """
        transformed = np.zeros_like(vertices)
        transformed[:, 0] = vertices[:, 0]   # hab_x = glb_x
        transformed[:, 1] = vertices[:, 2]   # hab_y = glb_z
        transformed[:, 2] = -vertices[:, 1]  # hab_z = -glb_y
        return transformed
    
    @staticmethod
    def transform_mesh_to_habitat(mesh):
        """
        Transform mesh vertices from GLB to Habitat coordinate system.
        
        Args:
            mesh: trimesh.Trimesh object in GLB coords
            
        Returns:
            trimesh.Trimesh object in Habitat coords
        """
        import trimesh
        
        transformed = mesh.copy()
        vertices = transformed.vertices.copy()
        
        # Apply coordinate transform: hab_x = glb_x, hab_y = glb_z, hab_z = -glb_y
        new_vertices = np.zeros_like(vertices)
        new_vertices[:, 0] = vertices[:, 0]   # hab_x = glb_x
        new_vertices[:, 1] = vertices[:, 2]   # hab_y = glb_z
        new_vertices[:, 2] = -vertices[:, 1]  # hab_z = -glb_y
        
        transformed.vertices = new_vertices
        return transformed
    
    def extract_object_mesh_by_aabb(
        self, 
        aabb_center: np.ndarray, 
        aabb_sizes: np.ndarray,
        padding: float = 0.05
    ):
        """
        Extract mesh faces within the given AABB (in Habitat coordinates).
        
        Args:
            aabb_center: Center of AABB in Habitat coordinates [x, y, z]
            aabb_sizes: Sizes of AABB in Habitat coordinates [sx, sy, sz]
            padding: Extra padding around AABB to catch edge vertices
            
        Returns:
            trimesh.Trimesh object containing the extracted mesh (in Habitat coords)
        """
        import trimesh
        
        self._load_meshes()
        
        # Convert AABB from Habitat to GLB coordinates
        glb_center = self.habitat_to_glb_coords(aabb_center)
        # Sizes also need coordinate swap: glb_sizes = [hab_sx, hab_sz, hab_sy]
        glb_sizes = np.array([aabb_sizes[0], aabb_sizes[2], aabb_sizes[1]])
        
        glb_min = glb_center - glb_sizes / 2 - padding
        glb_max = glb_center + glb_sizes / 2 + padding
        
        # Find meshes with vertices in this AABB
        all_extracted_meshes = []
        
        for mesh in self._meshes:
            if not isinstance(mesh, trimesh.Trimesh):
                continue
            
            vertices = mesh.vertices
            
            # Find vertices within AABB
            in_box = (
                (vertices[:, 0] >= glb_min[0]) & (vertices[:, 0] <= glb_max[0]) &
                (vertices[:, 1] >= glb_min[1]) & (vertices[:, 1] <= glb_max[1]) &
                (vertices[:, 2] >= glb_min[2]) & (vertices[:, 2] <= glb_max[2])
            )
            
            if np.sum(in_box) == 0:
                continue
            
            # Find faces where at least 2 vertices are in the box
            faces = mesh.faces
            vertices_in_face = in_box[faces]
            valid_faces = np.sum(vertices_in_face, axis=1) >= 2
            
            if not np.any(valid_faces):
                continue
            
            # Extract submesh
            extracted = mesh.submesh([np.where(valid_faces)[0]], append=True)
            all_extracted_meshes.append(extracted)
        
        if not all_extracted_meshes:
            raise ValueError(f"No mesh found in AABB: center={aabb_center}, sizes={aabb_sizes}")
        
        # Combine all extracted meshes
        combined = trimesh.util.concatenate(all_extracted_meshes)
        
        # Transform to Habitat coordinates
        combined = self.transform_mesh_to_habitat(combined)
        
        print(f"Extracted mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")
        
        return combined
    
    def extract_object_mesh_by_color(
        self,
        semantic_id: int,
        color_tolerance: int = 2,
        aabb_filter: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        aabb_padding: float = 0.3,
        min_vertices: int = 50
    ):
        """
        Extract object mesh based on vertex color matching.
        
        This method provides more precise extraction than AABB-based method by
        matching vertex colors to the semantic color defined in semantic.txt.
        
        Args:
            semantic_id: Semantic ID from semantic.txt
            color_tolerance: Maximum allowed color difference per channel (0-255)
            aabb_filter: Optional (center, sizes) tuple to filter by AABB first
            aabb_padding: Padding to add around AABB filter
            min_vertices: Minimum vertices to match (triggers adaptive tolerance)
            
        Returns:
            trimesh.Trimesh object containing the extracted mesh (in Habitat coords)
        """
        import trimesh
        
        self._load_meshes()
        
        # Get target color from semantic parser
        target_color = np.array(self.semantic_parser.get_color_by_id(semantic_id))
        
        # Find the best matching color and tolerance
        best_color, best_tolerance = self._find_best_color_tolerance(target_color, min_vertices)
        
        # Use provided tolerance if it's larger
        tolerance = max(color_tolerance, best_tolerance)
        
        print(f"  Target color: RGB{tuple(target_color)}, Using color: RGB{tuple(best_color.astype(int))}, Tolerance: {tolerance}")
        
        # Color matching: find vertices within tolerance
        color_diff = np.abs(self._vertex_colors.astype(np.float32) - best_color.astype(np.float32))
        max_diff = np.max(color_diff, axis=1)
        vertex_mask = max_diff <= tolerance
        
        # Optional AABB spatial filter
        if aabb_filter is not None:
            aabb_center, aabb_sizes = aabb_filter
            in_aabb = self._vertices_in_aabb_glb(aabb_center, aabb_sizes, aabb_padding)
            vertex_mask = vertex_mask & in_aabb
            print(f"  AABB filter: {np.sum(in_aabb)} vertices in AABB, {np.sum(vertex_mask)} after color+AABB")
        
        n_matched = np.sum(vertex_mask)
        print(f"  Matched {n_matched} vertices by color")
        
        if n_matched < 3:
            raise ValueError(
                f"Not enough vertices matched for semantic_id {semantic_id}: "
                f"only {n_matched} vertices found with color tolerance {tolerance}"
            )
        
        # Find faces where ALL THREE vertices match the color
        face_vertex_mask = vertex_mask[self._faces]  # (M, 3)
        face_mask = np.all(face_vertex_mask, axis=1)  # (M,) - all 3 vertices must match
        
        n_faces = np.sum(face_mask)
        print(f"  Matched {n_faces} faces (all 3 vertices match)")
        
        if n_faces == 0:
            # Fallback: if no complete faces, try faces with at least 2 matching vertices
            face_mask = np.sum(face_vertex_mask, axis=1) >= 2
            n_faces = np.sum(face_mask)
            print(f"  Fallback: {n_faces} faces with at least 2 matching vertices")
            
            if n_faces == 0:
                raise ValueError(
                    f"No faces found for semantic_id {semantic_id} with color tolerance {tolerance}"
                )
        
        # Extract matched faces
        matched_faces = self._faces[face_mask]
        
        # Re-index vertices (only keep used vertices)
        used_vertex_indices = np.unique(matched_faces.flatten())
        old_to_new = {old: new for new, old in enumerate(used_vertex_indices)}
        
        new_vertices = self._vertices[used_vertex_indices]
        new_faces = np.array([[old_to_new[v] for v in face] for face in matched_faces])
        
        # Transform vertices to Habitat coordinates
        new_vertices = self.glb_to_habitat_vertices(new_vertices)
        
        # Create trimesh object
        extracted_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False
        )
        
        print(f"  Extracted mesh: {len(extracted_mesh.vertices)} vertices, {len(extracted_mesh.faces)} faces")
        
        return extracted_mesh
    
    def save_object_mesh(self, mesh, output_path: str):
        """
        Save extracted mesh to an OBJ file (better compatibility with habitat-sim).
        
        Args:
            mesh: trimesh.Trimesh object to save
            output_path: Path to save the file
        """
        import trimesh
        
        # Ensure the mesh has a uniform color for rendering
        if hasattr(mesh.visual, 'vertex_colors'):
            mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 4), dtype=np.uint8) * 255
        
        # Save as OBJ for better habitat-sim compatibility
        mesh.export(output_path)
        print(f"Saved mesh to {output_path}")


# ============================================================================
# HM3DUnoccludedMaskGenerator - Core functionality class
# ============================================================================

class HM3DUnoccludedMaskGenerator:
    """
    Generate unoccluded masks, bboxes, and 3D boxes for HM3D objects.
    
    This class orchestrates the entire pipeline:
    1. Load the full HM3D scene to get semantic_scene with object AABBs
    2. Extract target object mesh based on AABB
    3. Transform mesh to Habitat coordinates
    4. Create a temporary scene with just the object
    5. Render from specified viewpoint
    6. Generate mask, bbox, and 3D box
    """
    
    def __init__(
        self, 
        scene_folder: str,
        data_path: str = SCENE_DATA_PATH,
        resolution: Tuple[int, int] = SENSOR_RESOLUTION,
        sensor_height: float = 1.0,
        hfov: float = 90.0,
        temp_dir: Optional[str] = MESH_TEMP_DIR,
        semantic_objects: Optional[Dict[int, Tuple[np.ndarray, np.ndarray, str]]] = None
    ):
        """
        Initialize the unoccluded mask generator.
        
        Args:
            scene_folder: Path to the HM3D scene folder (e.g., ".../00009-vLpv2VX547B")
            data_path: Path to habitat data directory
            resolution: Sensor resolution (height, width)
            sensor_height: Height of the sensor/camera
            hfov: Horizontal field of view in degrees
            temp_dir: Directory for temporary files (default: system temp)
            semantic_objects: Optional pre-extracted semantic objects dict 
                              {semantic_id: (center, sizes, category)}
                              If provided, skips loading scene to get AABBs
        """
        self.scene_folder = scene_folder
        self.data_path = data_path
        self.resolution = resolution
        self.sensor_height = sensor_height
        self.hfov = hfov
        self.temp_dir = temp_dir
        
        # Derive paths from scene folder
        self.scene_name = os.path.basename(scene_folder)
        if '-' in self.scene_name:
            self.scene_id = self.scene_name.split('-')[1]
        else:
            self.scene_id = self.scene_name
        
        self.semantic_txt_path = os.path.join(
            scene_folder, f"{self.scene_id}.semantic.txt"
        )
        self.semantic_glb_path = os.path.join(
            scene_folder, f"{self.scene_id}.semantic.glb"
        )
        self.basis_glb_path = os.path.join(
            scene_folder, f"{self.scene_id}.basis.glb"
        )
        
        # Validate paths
        if not os.path.exists(self.semantic_txt_path):
            raise FileNotFoundError(f"Semantic txt not found: {self.semantic_txt_path}")
        if not os.path.exists(self.semantic_glb_path):
            raise FileNotFoundError(f"Semantic glb not found: {self.semantic_glb_path}")
        
        # Initialize components
        self.parser = HM3DSemanticParser(self.semantic_txt_path)
        self.extractor = HM3DMeshExtractor(self.semantic_glb_path, self.parser)
        
        # Use pre-extracted semantic_objects or load from scene
        if semantic_objects is not None:
            self._semantic_objects = semantic_objects
            print(f"Using pre-extracted {len(self._semantic_objects)} semantic objects")
        else:
            # Load semantic scene to get object AABBs
            self._semantic_objects = {}  # semantic_id -> (center, sizes, category)
            self._load_semantic_scene()
        
        # Cache for extracted meshes
        self._mesh_cache: Dict[int, str] = {}  # semantic_id -> temp_obj_path
        
        # Cache for preprocessed metadata
        self._preprocessed_metadata: Optional[Dict] = None
    
    def _load_semantic_scene(self):
        """Load semantic scene to get object AABBs."""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = os.path.join(
            self.data_path, "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        )
        sim_cfg.scene_id = self.basis_glb_path
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = list(self.resolution)
        rgb_spec.position = [0.0, self.sensor_height, 0.0]
        agent_cfg.sensor_specifications = [rgb_spec]
        agent_cfg.action_space = {
            "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        }
        
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        
        try:
            semantic_scene = sim.semantic_scene
            for obj in semantic_scene.objects:
                if obj and obj.category and obj.aabb.sizes[0] > 0:
                    center = np.array(obj.aabb.center)
                    sizes = np.array(obj.aabb.sizes)
                    category = obj.category.name()
                    self._semantic_objects[obj.semantic_id] = (center, sizes, category)
            
            print(f"Loaded {len(self._semantic_objects)} semantic objects with valid AABBs")
        finally:
            sim.close()
    
    def _load_preprocessed_metadata(self) -> Optional[Dict]:
        """
        Load preprocessed metadata from object_mesh/metadata.json.
        
        Returns:
            Metadata dict or None if not found
        """
        if self._preprocessed_metadata is not None:
            return self._preprocessed_metadata
        
        metadata_path = os.path.join(self.scene_folder, "object_mesh", "metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                self._preprocessed_metadata = json.load(f)
            return self._preprocessed_metadata
        return None
    
    def _get_preprocessed_object_info(self, semantic_id: int) -> Optional[Dict]:
        """
        Get preprocessed object info for a semantic ID.
        
        Args:
            semantic_id: The semantic ID to look up
            
        Returns:
            Object info dict or None if not found
        """
        metadata = self._load_preprocessed_metadata()
        if metadata is None:
            return None
        
        for obj in metadata.get("valid_objects", []):
            if obj["semantic_id"] == semantic_id:
                return obj
        return None
    
    def _create_simulator(self, scene_path: str) -> habitat_sim.Simulator:
        """
        Create a habitat-sim simulator for the given scene.
        
        Args:
            scene_path: Path to the scene file (OBJ or GLB)
            
        Returns:
            habitat_sim.Simulator instance
        """
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        sensors = []
        
        # RGB sensor - use consistent sensor_height with original scene
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = list(self.resolution)
        rgb_spec.position = [0.0, self.sensor_height, 0.0]
        rgb_spec.hfov = mn.Deg(self.hfov)
        sensors.append(rgb_spec)
        
        # Depth sensor
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = list(self.resolution)
        depth_spec.position = [0.0, self.sensor_height, 0.0]
        depth_spec.hfov = mn.Deg(self.hfov)
        sensors.append(depth_spec)
        
        agent_cfg.sensor_specifications = sensors
        agent_cfg.action_space = {
            "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        }
        
        return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    
    def _get_or_create_object_mesh(
        self, 
        semantic_id: int,
        use_color_matching: bool = True,
        color_tolerance: int = 2
    ) -> str:
        """
        Get cached mesh path, load preprocessed mesh, or create new extracted mesh.
        
        Priority:
        1. Check memory cache
        2. Check preprocessed mesh (object_mesh/{semantic_id}.obj)
        3. Raise error if not found (no longer extract at runtime)
        
        Args:
            semantic_id: The semantic ID to extract
            use_color_matching: If True, use color-based extraction (more precise)
                               If False, use AABB-based extraction (faster but less precise)
            color_tolerance: Color tolerance for color matching (0-255)
            
        Returns:
            Path to the OBJ file containing the object mesh
        """
        cache_key = f"{semantic_id}_color" if use_color_matching else semantic_id
        
        # 1. Check memory cache
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        
        # 2. Check preprocessed mesh
        preprocessed_info = self._get_preprocessed_object_info(semantic_id)
        if preprocessed_info is not None:
            mesh_path = os.path.join(
                self.scene_folder, "object_mesh", preprocessed_info["mesh_file"]
            )
            if os.path.exists(mesh_path):
                # Load mesh_offset from preprocessed info
                mesh_offset = np.array(preprocessed_info["mesh_offset"])
                self._mesh_cache[f"{semantic_id}_offset"] = mesh_offset
                self._mesh_cache[cache_key] = mesh_path
                print(f"Using preprocessed mesh for semantic_id={semantic_id} ({preprocessed_info['category']})")
                return mesh_path
        
        # 3. Preprocessed mesh not found - raise error
        # We no longer extract meshes at runtime to avoid training crashes
        raise ValueError(
            f"Preprocessed mesh not found for semantic_id {semantic_id} in scene {self.scene_folder}. "
            f"Please run the preprocessing script first: "
            f"python dataset/preprocess_hm3d_meshes.py"
        )
    
    def _set_agent_state(
        self, 
        sim: habitat_sim.Simulator,
        position: np.ndarray,
        rotation: float,
        sensor_states: Optional[Dict] = None
    ):
        """
        Set the agent's position, rotation, and optionally sensor states.
        
        Args:
            sim: Simulator instance
            position: 3D position [x, y, z]
            rotation: Rotation angle in radians (around Y axis)
            sensor_states: Optional dict of sensor states from original agent.
                           Only 'rgb' and 'depth' sensor states will be synced
                           since this simulator only has these two sensors.
        """
        agent = sim.get_agent(0)
        agent_state = agent.get_state()
        agent_state.position = np.array(position, dtype=np.float32)
        
        rotation_quat = mn.Quaternion.rotation(
            mn.Rad(rotation), mn.Vector3(0, 1, 0)
        )
        agent_state.rotation = np.quaternion(
            rotation_quat.scalar,
            rotation_quat.vector.x,
            rotation_quat.vector.y,
            rotation_quat.vector.z
        )
        
        # Sync sensor states if provided (only rgb and depth)
        if sensor_states:
            synced_sensor_states = {}
            for sensor_name in ['rgb', 'depth']:
                if sensor_name in sensor_states:
                    sensor_pose = sensor_states[sensor_name]
                    synced_sensor_states[sensor_name] = SixDOFPose(
                        position=sensor_pose["position"],
                        rotation=sensor_pose["rotation"]
                    )
            if synced_sensor_states:
                agent_state.sensor_states = synced_sensor_states
        
        # NOTE: infer_sensor_states=False is important for sync sensor states 
        # (e.g. look_down, look_up)
        agent.set_state(agent_state, infer_sensor_states=False)
    
    def _compute_bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Compute bounding box from binary mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Tuple (xmin, xmax, ymin, ymax)
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        return (int(xmin), int(xmax), int(ymin), int(ymax))

    def _compute_min_area_rect(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute minimum area rectangle for 2D points using rotating calipers.

        Args:
            points: (N, 2) array of 2D points

        Returns:
            Tuple of (center, axes, extents)
            - center: (2,) array
            - axes: (2, 2) array, columns are axes
            - extents: (2,) array, half-sizes
        """
        if len(points) < 3:
            min_b = np.min(points, axis=0)
            max_b = np.max(points, axis=0)
            center = (min_b + max_b) / 2.0
            extents = (max_b - min_b) / 2.0
            return center, np.eye(2), extents

        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
        except Exception:
            # Fallback if ConvexHull fails (e.g. collinear points)
            min_b = np.min(points, axis=0)
            max_b = np.max(points, axis=0)
            center = (min_b + max_b) / 2.0
            extents = (max_b - min_b) / 2.0
            return center, np.eye(2), extents

        min_area = float('inf')
        best_center = points[0]
        best_axes = np.eye(2)
        best_extents = np.zeros(2)

        n = len(hull_points)
        for i in range(n):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % n]

            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-6: continue

            u = edge / norm
            v = np.array([-u[1], u[0]])

            axes = np.vstack([u, v]).T
            proj = points @ axes

            min_p = np.min(proj, axis=0)
            max_p = np.max(proj, axis=0)

            diff = max_p - min_p
            area = diff[0] * diff[1]

            if area < min_area:
                min_area = area
                best_axes = axes
                best_extents = diff / 2.0
                center_proj = (min_p + max_p) / 2.0
                best_center = axes @ center_proj

        return best_center, best_axes, best_extents
    
    def _compute_3d_bbox(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        agent_state
    ) -> Dict:
        """
        Compute 3D bounding box from depth and mask.
        
        Args:
            depth: Depth observation (H, W)
            mask: Binary mask (H, W)
            agent_state: Agent state for coordinate transformation
            
        Returns:
            Dictionary with 3D bbox information
        """
        H, W = depth.shape
        
        # Camera intrinsics
        focal_length = W / (2.0 * np.tan(np.radians(self.hfov) / 2.0))
        cx, cy = W / 2.0, H / 2.0
        
        # Get masked depth values
        v_coords, u_coords = np.where(mask)
        z_values = depth[v_coords, u_coords]
        
        # Filter invalid depths
        valid_mask = z_values > 0.01
        if not np.any(valid_mask):
            return self._create_default_3d_bbox()
        
        u_coords = u_coords[valid_mask]
        v_coords = v_coords[valid_mask]
        z_values = z_values[valid_mask]
        
        # Convert to 3D camera coordinates
        x_coords = (u_coords - cx) * z_values / focal_length
        y_coords = -(v_coords - cy) * z_values / focal_length
        z_coords = -z_values
        
        points_3d = np.stack([x_coords, y_coords, z_coords], axis=1)

        # Compute Global AABB (for reference and compatibility)
        min_bound = np.min(points_3d, axis=0)
        max_bound = np.max(points_3d, axis=0)

        # Compute OBB (Minimizing projection area on X-Z plane)
        # Habitat coordinates: Y is up, so horizontal plane is X-Z
        points_2d = points_3d[:, [0, 2]]  # X, Z
        center_2d, axes_2d, extents_2d = self._compute_min_area_rect(points_2d)

        # Vertical dimension (Y) is still axis-aligned
        min_y = np.min(points_3d[:, 1])
        max_y = np.max(points_3d[:, 1])
        center_y = (min_y + max_y) / 2.0
        extent_y = (max_y - min_y) / 2.0

        # Construct 3D OBB params
        center = np.array([center_2d[0], center_y, center_2d[1]])
        extents_3d = np.array([extents_2d[0], extent_y, extents_2d[1]])
        size = extents_3d * 2

        # OBB Axes (3x3 rotation matrix)
        obb_axes = np.eye(3)
        # X' axis -> axes_2d[:, 0]
        obb_axes[0, 0] = axes_2d[0, 0]
        obb_axes[0, 2] = axes_2d[0, 1]
        # Y' axis -> (0, 1, 0)
        obb_axes[1, 1] = 1.0
        # Z' axis -> axes_2d[:, 1]
        obb_axes[2, 0] = axes_2d[1, 0]
        obb_axes[2, 2] = axes_2d[1, 1]

        # Compute 8 corners of the OBB
        # Order must match visualization utility expectation (bottom face, then top face, cycling CCW or CW)
        # Based on habitat_3dbox_utils.py, the order is:
        # 0: (-1, -1, -1) -> min_x, min_y, min_z
        # 1: ( 1, -1, -1) -> max_x, min_y, min_z
        # 2: ( 1,  1, -1) -> max_x, max_y, min_z
        # 3: (-1,  1, -1) -> min_x, max_y, min_z
        # 4: (-1, -1,  1) -> min_x, min_y, max_z
        # 5: ( 1, -1,  1) -> max_x, min_y, max_z
        # 6: ( 1,  1,  1) -> max_x, max_y, max_z
        # 7: (-1,  1,  1) -> min_x, max_y, max_z
        
        corner_signs = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ])

        corners = []
        for sign in corner_signs:
            offset = (
                sign[0] * obb_axes[:, 0] * extents_3d[0] +
                sign[1] * obb_axes[:, 1] * extents_3d[1] +
                sign[2] * obb_axes[:, 2] * extents_3d[2]
            )
            corners.append(center + offset)
        corners = np.array(corners)

        return {
            'center': center,
            'size': size,
            'min_bound': min_bound,
            'max_bound': max_bound,
            'corners': corners,
            'obb_center': center,
            'obb_axes': obb_axes,
            'obb_extents': extents_3d,
            'obb_corners': corners,
            'num_points': len(points_3d)
        }
    
    def _create_default_3d_bbox(self) -> Dict:
        """Create a default 3D bbox when detection fails."""
        center = np.array([0.0, 0.0, -2.0])
        size = np.array([0.1, 0.1, 0.1])
        half_size = size / 2.0
        min_bound = center - half_size
        max_bound = center + half_size
        
        corners = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
        ])
        
        return {
            'center': center,
            'size': size,
            'min_bound': min_bound,
            'max_bound': max_bound,
            'corners': corners,
            'obb_center': center,
            'obb_axes': np.eye(3),
            'obb_extents': half_size,
            'obb_corners': corners,
            'num_points': 0
        }
    
    def get_object_aabb(self, semantic_id: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get the AABB of an object in Habitat world coordinates.
        
        Args:
            semantic_id: Semantic ID of the target object
            
        Returns:
            Tuple of (center, sizes, category)
        """
        if semantic_id not in self._semantic_objects:
            raise ValueError(f"Semantic ID {semantic_id} not found in scene")
        return self._semantic_objects[semantic_id]
    
    def get_unoccluded_mask(
        self,
        semantic_id: int,
        agent_position: Union[np.ndarray, List[float]],
        agent_rotation: float,
        sensor_states: Optional[Dict] = None
    ) -> Dict:
        """
        Get unoccluded mask, bbox, and 3D box for a specific object.
        
        Args:
            semantic_id: Semantic ID of the target object
            agent_position: Agent position [x, y, z] in world coordinates
            agent_rotation: Agent rotation in radians (around Y axis)
            sensor_states: Optional dict of sensor states from original agent.
                           Used to sync sensor orientation (e.g., look_down/look_up).
            
        Returns:
            Dictionary containing:
                - bbox_gt: (xmin, xmax, ymin, ymax)
                - mask_gt: RLE encoded mask
                - bbox_3d_gt: 3D bounding box information
                - category: Object category name
                - rgb: RGB observation (PIL Image)
        """
        from habitat_sim.utils import viz_utils as vut
        
        # Get or create the object mesh
        object_obj_path = self._get_or_create_object_mesh(semantic_id)
        mesh_offset = self._mesh_cache.get(f"{semantic_id}_offset", np.zeros(3))
        
        # Create simulator for this object
        sim = self._create_simulator(object_obj_path)
        
        try:
            # Adjust agent position relative to centered mesh
            # The mesh was centered by subtracting mesh_offset from vertices
            # So agent position should also be adjusted by subtracting mesh_offset
            # This preserves the relative position between agent and object
            adjusted_position = np.array(agent_position, dtype=np.float32) - mesh_offset
            
            # Also adjust sensor_states positions by the same offset
            adjusted_sensor_states = None
            if sensor_states:
                adjusted_sensor_states = {}
                for sensor_name, sensor_pose in sensor_states.items():
                    adjusted_sensor_states[sensor_name] = {
                        "position": np.array(sensor_pose["position"], dtype=np.float32) - mesh_offset,
                        "rotation": sensor_pose["rotation"]
                    }
            
            self._set_agent_state(sim, adjusted_position, agent_rotation, adjusted_sensor_states)
            
            # Get observations
            obs = sim.get_sensor_observations()
            
            # Create mask from RGB (non-black pixels)
            rgb = obs["rgb"]
            mask = np.any(rgb[:, :, :3] > 0, axis=2).astype(np.uint8)
            
            # Check if object is visible
            if not np.any(mask):
                print(f"Warning: Object {semantic_id} not visible from this viewpoint")
                return {
                    'bbox_gt': (0, 0, 0, 0),
                    'mask_gt': mask_utils.encode(np.asfortranarray(mask)),
                    'bbox_3d_gt': self._create_default_3d_bbox(),
                    'category': self.parser.get_category_by_id(semantic_id),
                    'rgb': vut.observation_to_image(obs["rgb"], "color").convert("RGB")
                }
            
            # Compute bbox
            bbox_gt = self._compute_bbox_from_mask(mask)
            
            # Encode mask to RLE
            mask_rle = mask_utils.encode(np.asfortranarray(mask))
            if isinstance(mask_rle['counts'], bytes):
                mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            
            # Compute 3D bbox
            depth = obs["depth"]
            agent_state = sim.get_agent(0).get_state()
            bbox_3d_gt = self._compute_3d_bbox(depth, mask, agent_state)
            
            # Get RGB image
            rgb_pil = vut.observation_to_image(obs["rgb"], "color").convert("RGB")
            
            return {
                'bbox_gt': bbox_gt,
                'mask_gt': mask_rle,
                'bbox_3d_gt': bbox_3d_gt,
                'category': self.parser.get_category_by_id(semantic_id),
                'rgb': rgb_pil
            }
            
        finally:
            sim.close()
    
    def get_unoccluded_mask_batch(
        self,
        semantic_ids: List[int],
        agent_position: Union[np.ndarray, List[float]],
        agent_rotation: float,
        sensor_states: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get unoccluded masks for multiple objects.
        
        Args:
            semantic_ids: List of semantic IDs
            agent_position: Agent position [x, y, z]
            agent_rotation: Agent rotation in radians
            sensor_states: Optional dict of sensor states from original agent.
                           Used to sync sensor orientation (e.g., look_down/look_up).
            
        Returns:
            List of result dictionaries, one per semantic_id
        """
        results = []
        for semantic_id in semantic_ids:
            try:
                result = self.get_unoccluded_mask(
                    semantic_id, agent_position, agent_rotation, sensor_states
                )
                result['semantic_id'] = semantic_id
                results.append(result)
            except Exception as e:
                print(f"Error processing semantic_id {semantic_id}: {e}")
                continue
        return results
    
    def cleanup(self):
        """Clean up temporary files (but NOT preprocessed meshes)."""
        # Only clear the memory cache, don't delete preprocessed mesh files
        # Preprocessed meshes are in {scene_folder}/object_mesh/ and should be kept
        self._mesh_cache.clear()
    
    def __del__(self):
        """Destructor to clean up resources."""
        try:
            self.cleanup()
        except:
            pass


# ============================================================================
# Utility functions
# ============================================================================

def visualize_result(result: Dict, save_path: Optional[str] = None):
    """
    Visualize the unoccluded mask result.
    
    Args:
        result: Result dictionary from get_unoccluded_mask
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB with bbox
    rgb = result['rgb'].copy()
    draw = ImageDraw.Draw(rgb)
    xmin, xmax, ymin, ymax = result['bbox_gt']
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=3)
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB with BBox\nCategory: {result['category']}")
    axes[0].axis('off')
    
    # Mask
    mask = mask_utils.decode(result['mask_gt'])
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Unoccluded Mask")
    axes[1].axis('off')
    
    # Overlay
    rgb_np = np.array(result['rgb'])
    overlay = rgb_np.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red overlay
    blended = (0.6 * rgb_np + 0.4 * overlay).astype(np.uint8)
    axes[2].imshow(blended)
    axes[2].set_title("Mask Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


# ============================================================================
# Main test function
# ============================================================================

def render_original_scene(
    scene_folder: str,
    data_path: str,
    agent_position: np.ndarray,
    agent_rotation: float,
    semantic_id: int,
    resolution: Tuple[int, int] = (640, 800),
    sensor_height: float = 1.0,
    hfov: float = 90.0
) -> Tuple:
    """
    Render the original HM3D scene from a given viewpoint.
    
    Returns:
        Tuple of (rgb_pil, semantic_obs, depth_obs, visible_mask)
    """
    from habitat_sim.utils import viz_utils as vut
    
    scene_name = os.path.basename(scene_folder)
    if '-' in scene_name:
        scene_id = scene_name.split('-')[1]
    else:
        scene_id = scene_name
    
    basis_glb_path = os.path.join(scene_folder, f"{scene_id}.basis.glb")
    
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.join(
        data_path, "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    )
    sim_cfg.scene_id = basis_glb_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sensors = []
    
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = list(resolution)
    rgb_spec.position = [0.0, sensor_height, 0.0]
    rgb_spec.hfov = mn.Deg(hfov)
    sensors.append(rgb_spec)
    
    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = list(resolution)
    semantic_spec.position = [0.0, sensor_height, 0.0]
    semantic_spec.hfov = mn.Deg(hfov)
    sensors.append(semantic_spec)
    
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = list(resolution)
    depth_spec.position = [0.0, sensor_height, 0.0]
    depth_spec.hfov = mn.Deg(hfov)
    sensors.append(depth_spec)
    
    agent_cfg.sensor_specifications = sensors
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
    }
    
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    
    try:
        agent = sim.get_agent(0)
        agent_state = agent.get_state()
        agent_state.position = np.array(agent_position, dtype=np.float32)
        
        rotation_quat = mn.Quaternion.rotation(
            mn.Rad(agent_rotation), mn.Vector3(0, 1, 0)
        )
        agent_state.rotation = np.quaternion(
            rotation_quat.scalar,
            rotation_quat.vector.x,
            rotation_quat.vector.y,
            rotation_quat.vector.z
        )
        agent.set_state(agent_state)
        
        obs = sim.get_sensor_observations()
        
        rgb_pil = vut.observation_to_image(obs["rgb"], "color").convert("RGB")
        semantic_obs = obs["semantic"]
        depth_obs = obs["depth"]
        
        # Get mask for the target object (with occlusion)
        visible_mask = (semantic_obs == semantic_id).astype(np.uint8)
        
        return rgb_pil, semantic_obs, depth_obs, visible_mask
        
    finally:
        sim.close()


def main():
    """Test the HM3D unoccluded mask generation."""
    
    # Test scene path
    scene_folder = f"{SCENE_DATA_PATH}/hm3d/train/00009-vLpv2VX547B"
    data_path = SCENE_DATA_PATH
    
    print("=" * 60)
    print("HM3D Unoccluded Mask Generator Test")
    print("=" * 60)
    
    # Initialize generator
    print("\n1. Initializing generator...")
    generator = HM3DUnoccludedMaskGenerator(scene_folder, temp_dir=TEMP_DIR)
   
    # List some categories
    print("\n2. Available categories:")
    categories = generator.parser.get_all_categories()
    print(f"   Found {len(categories)} categories")
    print(f"   Sample categories: {categories[:10]}")
    
    # Find a chair
    chair_ids = generator.parser.get_ids_by_category("chair")
    print(f"\n3. Found {len(chair_ids)} chairs in semantic.txt: {chair_ids}")
    
    # Filter to chairs with valid AABBs in semantic_scene
    valid_chair_ids = [cid for cid in chair_ids if cid in generator._semantic_objects]
    print(f"   Chairs with valid AABBs: {valid_chair_ids}")
    
    if valid_chair_ids:
        # Test with first valid chair
        test_id = valid_chair_ids[1]
        print(f"\n4. Testing with semantic_id={test_id}")
        
        try:
            # Get object AABB
            center, sizes, category = generator.get_object_aabb(test_id)
            print(f"   Category: {category}")
            print(f"   AABB center: {center}")
            print(f"   AABB sizes: {sizes}")
            
            # Position agent to look at the object
            # Stand 2m in front of the object (in -Z direction from object)
            agent_pos = center.copy()
            agent_pos[2] += 1.0  # Move back in Z
            agent_pos[1] = 0.0   # Ground level
            
            # Face the object (look in -Z direction)
            agent_rotation = 0.0
            
            print(f"\n   Agent position: {agent_pos}")
            print(f"   Agent rotation: {agent_rotation}")
            
            # Get original scene rendering (with occlusion)
            print("\n5. Rendering original scene...")
            orig_rgb, orig_semantic, orig_depth, orig_mask = render_original_scene(
                scene_folder, data_path, agent_pos, agent_rotation, test_id
            )
            orig_rgb.save(os.path.join(TEMP_DIR, "hm3d_mask_test_original_rgb.png"))
            print(f"   Saved original scene RGB to: hm3d_mask_test_original_rgb.png")
            print(f"   Visible pixels in original scene: {np.sum(orig_mask)}")
            
            # Get unoccluded mask
            print("\n6. Getting unoccluded mask...")
            result = generator.get_unoccluded_mask(
                test_id, agent_pos, agent_rotation
            )
            
            print(f"\n7. Results:")
            print(f"   Category: {result['category']}")
            print(f"   BBox (unoccluded): {result['bbox_gt']}")
            print(f"   Mask size: {result['mask_gt']['size']}")
            
            # Decode unoccluded mask for comparison
            unoccluded_mask = mask_utils.decode(result['mask_gt'])
            print(f"   Unoccluded mask pixels: {np.sum(unoccluded_mask)}")
            print(f"   Visible mask pixels: {np.sum(orig_mask)}")
            
            if result['bbox_3d_gt']['num_points'] > 0:
                print(f"   3D bbox center: {result['bbox_3d_gt']['center']}")
                print(f"   3D bbox size: {result['bbox_3d_gt']['size']}")
            
            # Save unoccluded RGB image
            output_path = os.path.join(TEMP_DIR, "hm3d_mask_test_output.png")
            result['rgb'].save(output_path.replace('.png', '_unoccluded_rgb.png'))
            print(f"\n   Saved unoccluded RGB to: hm3d_mask_test_output_unoccluded_rgb.png")
            
            # Visualize comparison
            visualize_comparison(orig_rgb, orig_mask, result, save_path=output_path)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No valid chairs found with AABBs!")
    
    # Cleanup
    generator.cleanup()
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


def visualize_comparison(orig_rgb, orig_mask, unoccluded_result: Dict, save_path: Optional[str] = None):
    """
    Visualize comparison between original (occluded) and unoccluded views.
    """
    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 1: Original scene (with occlusion)
    # Original RGB
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title(f"Original Scene RGB\n(with occlusion)")
    axes[0, 0].axis('off')
    
    # Original mask (visible part only)
    axes[0, 1].imshow(orig_mask, cmap='gray')
    axes[0, 1].set_title(f"Visible Mask\n({np.sum(orig_mask)} pixels)")
    axes[0, 1].axis('off')
    
    # Original overlay
    orig_np = np.array(orig_rgb)
    orig_overlay = orig_np.copy()
    orig_overlay[orig_mask > 0] = [0, 255, 0]  # Green overlay
    blended_orig = (0.6 * orig_np + 0.4 * orig_overlay).astype(np.uint8)
    axes[0, 2].imshow(blended_orig)
    axes[0, 2].set_title("Original Mask Overlay")
    axes[0, 2].axis('off')
    
    # Row 2: Unoccluded view
    # Unoccluded RGB
    unoccluded_rgb = unoccluded_result['rgb']
    axes[1, 0].imshow(unoccluded_rgb)
    axes[1, 0].set_title(f"Unoccluded RGB\nCategory: {unoccluded_result['category']}")
    axes[1, 0].axis('off')
    
    # Unoccluded mask
    unoccluded_mask = mask_utils.decode(unoccluded_result['mask_gt'])
    axes[1, 1].imshow(unoccluded_mask, cmap='gray')
    axes[1, 1].set_title(f"Unoccluded Mask\n({np.sum(unoccluded_mask)} pixels)")
    axes[1, 1].axis('off')
    
    # Unoccluded overlay with bbox
    unocc_rgb = unoccluded_result['rgb'].copy()
    draw = ImageDraw.Draw(unocc_rgb)
    xmin, xmax, ymin, ymax = unoccluded_result['bbox_gt']
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=3)
    
    unocc_np = np.array(unocc_rgb)
    unocc_overlay = unocc_np.copy()
    unocc_overlay[unoccluded_mask > 0] = [255, 0, 0]  # Red overlay
    blended_unocc = (0.6 * unocc_np + 0.4 * unocc_overlay).astype(np.uint8)
    axes[1, 2].imshow(blended_unocc)
    axes[1, 2].set_title(f"Unoccluded Mask + BBox\nBBox: {unoccluded_result['bbox_gt']}")
    axes[1, 2].axis('off')
    
    # Column 4: Mask comparison on blank canvas
    H, W = orig_mask.shape
    
    # Create RGB image showing both masks
    # Green: visible mask (original scene)
    # Red: unoccluded mask  
    # Yellow: overlap (both masks)
    mask_comparison = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Unoccluded mask in red channel
    mask_comparison[:, :, 0] = (unoccluded_mask > 0).astype(np.uint8) * 255
    # Visible mask in green channel
    mask_comparison[:, :, 1] = (orig_mask > 0).astype(np.uint8) * 255
    
    axes[0, 3].imshow(mask_comparison)
    axes[0, 3].set_title("Mask Comparison\nRed: Unoccluded only\nGreen: Visible only\nYellow: Overlap")
    axes[0, 3].axis('off')
    
    # Difference visualization
    # Show only the difference (unoccluded - visible = occluded parts)
    diff_mask = np.zeros((H, W, 3), dtype=np.uint8)
    only_unoccluded = (unoccluded_mask > 0) & (orig_mask == 0)  # Parts only in unoccluded
    only_visible = (orig_mask > 0) & (unoccluded_mask == 0)     # Parts only in visible (shouldn't happen ideally)
    overlap = (unoccluded_mask > 0) & (orig_mask > 0)           # Overlapping parts
    
    diff_mask[only_unoccluded] = [255, 0, 0]    # Red: occluded parts
    diff_mask[only_visible] = [0, 255, 0]       # Green: visible only (error)
    diff_mask[overlap] = [255, 255, 255]        # White: visible in both
    
    occluded_pixels = np.sum(only_unoccluded)
    visible_only_pixels = np.sum(only_visible)
    overlap_pixels = np.sum(overlap)
    
    axes[1, 3].imshow(diff_mask)
    axes[1, 3].set_title(f"Mask Difference\nWhite: Visible ({overlap_pixels}px)\n"
                         f"Red: Occluded ({occluded_pixels}px)\n"
                         f"Green: Visible-only ({visible_only_pixels}px)")
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison visualization to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    main()

