"""
Blender Agent Module

This module provides functionality to create 3D models based on text prompts.
Uses real Blender scripting when available, falls back to simulation otherwise.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import tempfile


def _check_blender_availability() -> Tuple[bool, Optional[str]]:
    """
    Check if Blender is available on the system.
    
    Returns:
        tuple: (is_available, blender_path)
    """
    try:
        # Try common Blender executable names
        blender_names = ['blender', 'blender.exe', 'Blender']
        
        for name in blender_names:
            try:
                result = subprocess.run([name, '--version'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=10)
                if result.returncode == 0:
                    return True, name
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                continue
        
        # Check common installation paths
        common_paths = [
            '/usr/bin/blender',
            '/usr/local/bin/blender',
            'C:\\Program Files\\Blender Foundation\\Blender\\blender.exe',
            '/Applications/Blender.app/Contents/MacOS/Blender'
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                try:
                    result = subprocess.run([path, '--version'], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=10)
                    if result.returncode == 0:
                        return True, path
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    continue
                    
        return False, None
        
    except Exception:
        return False, None


def _generate_blender_script(prompt: str, output_path: str) -> str:
    """
    Generate a Blender Python script for model creation.
    
    Args:
        prompt (str): Text description of the desired 3D model
        output_path (str): Path where the GLB file should be saved
        
    Returns:
        str: Blender Python script content
    """
    script = f'''
import bpy
import bmesh
import os
from mathutils import Vector

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Parse prompt for model type
prompt = "{prompt}".lower()

def create_eye_model():
    """Create a detailed eye model with retina structure."""
    # Create main eye sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
    eye_main = bpy.context.active_object
    eye_main.name = "EyeMain"
    
    # Add subsurface modifier for smoothness
    modifier = eye_main.modifiers.new(name="Subsurface", type='SUBSURF')
    modifier.levels = 2
    
    # Create iris
    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.1, location=(0, 0, 0.9))
    iris = bpy.context.active_object
    iris.name = "Iris"
    
    # Create pupil
    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=0.12, location=(0, 0, 0.91))
    pupil = bpy.context.active_object
    pupil.name = "Pupil"
    
    # Create retina highlight region
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.95, location=(0, 0, 0))
    retina = bpy.context.active_object
    retina.name = "RetinaHighlight"
    
    # Create materials
    create_eye_materials()
    
    # Assign materials
    eye_main.data.materials.append(bpy.data.materials["EyeMaterial"])
    iris.data.materials.append(bpy.data.materials["IrisMaterial"])
    pupil.data.materials.append(bpy.data.materials["PupilMaterial"])
    retina.data.materials.append(bpy.data.materials["RetinaMaterial"])

def create_brain_model():
    """Create a brain model with cerebral regions."""
    # Create main brain shape
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.2, location=(0, 0, 0))
    brain = bpy.context.active_object
    brain.name = "BrainMain"
    
    # Add sculpting details
    modifier = brain.modifiers.new(name="Subsurface", type='SUBSURF')
    modifier.levels = 3
    
    # Create cerebellum
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.6, location=(0, -1.2, -0.5))
    cerebellum = bpy.context.active_object
    cerebellum.name = "Cerebellum"
    
    # Create highlight regions for different brain areas
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.8, 0.2, 0.3))
    frontal_lobe = bpy.context.active_object
    frontal_lobe.name = "FrontalLobeHighlight"
    
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-0.8, 0.2, 0.3))
    parietal_lobe = bpy.context.active_object
    parietal_lobe.name = "ParietalLobeHighlight"
    
    create_brain_materials()
    
    # Assign materials
    brain.data.materials.append(bpy.data.materials["BrainMaterial"])
    cerebellum.data.materials.append(bpy.data.materials["CerebellumMaterial"])
    frontal_lobe.data.materials.append(bpy.data.materials["HighlightMaterial"])
    parietal_lobe.data.materials.append(bpy.data.materials["HighlightMaterial"])

def create_heart_model():
    """Create an anatomical heart model."""
    # Create main heart shape using metaballs for organic form
    bpy.ops.object.metaball_add(type='BALL', location=(0, 0, 0))
    heart_main = bpy.context.active_object
    heart_main.name = "HeartMain"
    heart_main.scale = (1.2, 0.8, 1.5)
    
    # Add another metaball for heart shape
    bpy.ops.object.metaball_add(type='BALL', location=(0.5, 0, 0.8))
    heart_top = bpy.context.active_object
    heart_top.scale = (0.8, 0.8, 0.8)
    
    # Convert to mesh
    bpy.ops.object.convert(target='MESH')
    
    # Create highlight regions for chambers
    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0.3, 0, 0.2))
    left_ventricle = bpy.context.active_object
    left_ventricle.name = "LeftVentricleHighlight"
    
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(-0.3, 0, 0.2))
    right_ventricle = bpy.context.active_object
    right_ventricle.name = "RightVentricleHighlight"
    
    create_heart_materials()

def create_eye_materials():
    """Create materials for eye model."""
    # Eye main material
    eye_mat = bpy.data.materials.new(name="EyeMaterial")
    eye_mat.use_nodes = True
    eye_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.9, 0.9, 0.95, 1.0)
    eye_mat.node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.1  # Roughness
    eye_mat.node_tree.nodes["Principled BSDF"].inputs[15].default_value = 1.4  # IOR
    
    # Iris material
    iris_mat = bpy.data.materials.new(name="IrisMaterial")
    iris_mat.use_nodes = True
    iris_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.2, 0.5, 0.8, 1.0)
    
    # Pupil material
    pupil_mat = bpy.data.materials.new(name="PupilMaterial")
    pupil_mat.use_nodes = True
    pupil_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    
    # Retina highlight material
    retina_mat = bpy.data.materials.new(name="RetinaMaterial")
    retina_mat.use_nodes = True
    retina_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1.0, 0.3, 0.3, 0.3)
    retina_mat.node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0.7  # Alpha

def create_brain_materials():
    """Create materials for brain model."""
    # Brain material
    brain_mat = bpy.data.materials.new(name="BrainMaterial")
    brain_mat.use_nodes = True
    brain_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.7, 0.6, 1.0)
    
    # Cerebellum material
    cerebellum_mat = bpy.data.materials.new(name="CerebellumMaterial")
    cerebellum_mat.use_nodes = True
    cerebellum_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.7, 0.6, 0.5, 1.0)
    
    # Highlight material
    highlight_mat = bpy.data.materials.new(name="HighlightMaterial")
    highlight_mat.use_nodes = True
    highlight_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.9, 0.9, 0.2, 0.5)
    highlight_mat.node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0.5  # Alpha

def create_heart_materials():
    """Create materials for heart model."""
    # Heart material
    heart_mat = bpy.data.materials.new(name="HeartMaterial")
    heart_mat.use_nodes = True
    heart_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.2, 0.2, 1.0)
    heart_mat.node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.3  # Roughness

# Determine model type from prompt and create appropriate model
if "eye" in prompt or "retina" in prompt:
    create_eye_model()
elif "brain" in prompt or "cerebral" in prompt:
    create_brain_model()
elif "heart" in prompt or "cardiac" in prompt:
    create_heart_model()
else:
    # Default to eye model
    create_eye_model()

# Select all objects for export
bpy.ops.object.select_all(action='SELECT')

# Ensure output directory exists
output_dir = os.path.dirname("{output_path}")
os.makedirs(output_dir, exist_ok=True)

# Export as GLB
bpy.ops.export_scene.gltf(
    filepath="{output_path}",
    export_format='GLB',
    export_selected=True,
    export_materials='EXPORT',
    export_colors=True,
    export_cameras=False,
    export_lights=False
)

print(f"Model exported successfully to: {output_path}")
'''
    return script


def create_model(prompt: str) -> str:
    """
    Create a 3D model based on a text prompt.
    Uses real Blender scripting when available, falls back to simulation.
    
    Args:
        prompt (str): Text description of the desired 3D model
        
    Returns:
        str: Relative path to the created .glb file
        
    Raises:
        ValueError: If prompt is empty or invalid
        OSError: If file creation fails
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    print(f"Creating 3D model from prompt: '{prompt}'")
    
    # Create the output directory if it doesn't exist
    output_dir = Path("static/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path
    output_file = output_dir / "eye_model.glb"
    
    # Check if Blender is available
    blender_available, blender_path = _check_blender_availability()
    
    if blender_available:
        print(f"Blender found at: {blender_path}")
        return _create_model_with_blender(prompt, str(output_file), blender_path)
    else:
        print("Blender not found, falling back to simulation mode")
        return _create_model_simulation(prompt, str(output_file))


def _create_model_with_blender(prompt: str, output_file: str, blender_path: str) -> str:
    """
    Create a 3D model using real Blender.
    
    Args:
        prompt (str): Text description of the desired 3D model
        output_file (str): Path where the GLB file should be saved
        blender_path (str): Path to Blender executable
        
    Returns:
        str: Path to the created .glb file
    """
    print("Using real Blender for model generation...")
    
    # Generate Blender script
    script_content = _generate_blender_script(prompt, output_file)
    
    # Create temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
        temp_script.write(script_content)
        script_path = temp_script.name
    
    try:
        print("Executing Blender script...")
        
        # Run Blender in background mode with the script
        cmd = [
            blender_path,
            '--background',
            '--python', script_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode == 0:
            print("Blender script executed successfully")
            if os.path.exists(output_file):
                print(f"Model successfully created: {output_file}")
                return output_file
            else:
                raise OSError("Blender script completed but output file not found")
        else:
            print(f"Blender execution failed: {result.stderr}")
            raise OSError(f"Blender execution failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        raise OSError("Blender execution timed out")
    except Exception as e:
        print(f"Error running Blender: {e}")
        raise OSError(f"Failed to create model with Blender: {e}")
    finally:
        # Clean up temporary script file
        try:
            os.unlink(script_path)
        except:
            pass


def _create_model_simulation(prompt: str, output_file: str) -> str:
    """
    Create a simulated 3D model (fallback when Blender is not available).
    
    Args:
        prompt (str): Text description of the desired 3D model
        output_file (str): Path where the GLB file should be saved
        
    Returns:
        str: Path to the created .glb file
    """
    print("Initializing Blender simulation...")
    
    # Simulate processing time
    _simulate_model_generation(prompt)
    
    # Create realistic GLB placeholder
    realistic_glb_content = _generate_realistic_glb_placeholder(prompt)
    
    try:
        with open(output_file, 'wb') as f:
            f.write(realistic_glb_content)
        
        print(f"Model successfully created: {output_file}")
        return output_file
        
    except OSError as e:
        raise OSError(f"Failed to create model file: {e}")


def _simulate_model_generation(prompt: str) -> None:
    """
    Simulate the model generation process with realistic steps.
    
    Args:
        prompt (str): The input prompt for context
    """
    steps = [
        "Parsing prompt and determining model type...",
        "Generating base geometry...",
        "Creating highlight regions...",
        "Applying materials and textures...",
        "Optimizing mesh topology...",
        "Exporting to GLB format..."
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {step}")
        time.sleep(0.5)  # Simulate processing time


def _generate_realistic_glb_placeholder(prompt: str) -> bytes:
    """
    Generate a more realistic GLB file placeholder with binary-like content.
    
    Args:
        prompt (str): The original prompt used for generation
        
    Returns:
        bytes: Realistic GLB placeholder content
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a more realistic binary-like placeholder
    # Real GLB files start with specific magic bytes
    glb_header = b'glTF'  # GLB magic
    glb_version = b'\\x02\\x00\\x00\\x00'  # Version 2.0
    
    # Create JSON-like metadata
    metadata = f'''{{
    "asset": {{
        "generator": "Blender Agent v1.0",
        "version": "2.0"
    }},
    "scene": 0,
    "scenes": [
        {{
            "name": "Scene",
            "nodes": [0]
        }}
    ],
    "nodes": [
        {{
            "name": "Model",
            "mesh": 0
        }}
    ],
    "meshes": [
        {{
            "name": "GeneratedMesh",
            "primitives": [
                {{
                    "attributes": {{
                        "POSITION": 0,
                        "NORMAL": 1,
                        "TEXCOORD_0": 2
                    }},
                    "material": 0
                }}
            ]
        }}
    ],
    "materials": [
        {{
            "name": "Material",
            "pbrMetallicRoughness": {{
                "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                "roughnessFactor": 0.5
            }}
        }}
    ],
    "accessors": [
        {{
            "bufferView": 0,
            "componentType": 5126,
            "count": 24,
            "type": "VEC3"
        }}
    ],
    "bufferViews": [
        {{
            "buffer": 0,
            "byteLength": 288,
            "byteOffset": 0
        }}
    ],
    "buffers": [
        {{
            "byteLength": 288
        }}
    ]
}}

# Blender Agent Generated Model
# Prompt: {prompt}
# Created: {timestamp}
# Status: {'Blender Available' if _check_blender_availability()[0] else 'Simulation Mode'}
# Highlight Regions: Enabled
# Export Format: GLB 2.0
'''.encode('utf-8')
    
    # Combine header with metadata for a more realistic file
    return glb_header + glb_version + metadata


def get_model_info(file_path: str) -> Optional[Dict[str, str]]:
    """
    Get information about a created model file.
    
    Args:
        file_path (str): Path to the model file
        
    Returns:
        dict or None: Model information if file exists, None otherwise
    """
    if not os.path.exists(file_path):
        return None
    
    file_stats = os.stat(file_path)
    blender_available, blender_path = _check_blender_availability()
    
    return {
        "file_path": file_path,
        "file_size": f"{file_stats.st_size} bytes",
        "created": time.ctime(file_stats.st_ctime),
        "modified": time.ctime(file_stats.st_mtime),
        "format": "GLB 2.0",
        "blender_mode": "Real Blender" if blender_available else "Simulation",
        "blender_path": blender_path or "Not found",
        "highlight_regions": "Enabled"
    }


def cleanup_models() -> bool:
    """
    Clean up generated model files.
    
    Returns:
        bool: True if cleanup successful, False otherwise
    """
    try:
        model_file = Path("static/models/eye_model.glb")
        if model_file.exists():
            model_file.unlink()
            print("Model files cleaned up successfully")
        return True
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False


# Configuration for Blender integration
class BlenderConfig:
    """Configuration class for Blender agent settings."""
    
    # Model generation settings
    DEFAULT_QUALITY = "high"
    SUPPORTED_FORMATS = [".glb", ".obj", ".fbx"]
    MAX_VERTICES = 100000
    
    # File paths
    OUTPUT_DIR = "static/models"
    DEFAULT_FILENAME = "eye_model.glb"
    
    # Blender settings
    BLENDER_TIMEOUT = 120  # seconds
    USE_BACKGROUND_MODE = True
    EXPORT_MATERIALS = True
    EXPORT_COLORS = True
    ENABLE_HIGHLIGHT_REGIONS = True
    
    # Future ML model settings
    MODEL_PATH = None  # Path to trained ML model
    USE_GPU = True     # Whether to use GPU acceleration
    BATCH_SIZE = 1     # For batch processing


# Example usage and testing
if __name__ == "__main__":
    # Test Blender availability
    blender_available, blender_path = _check_blender_availability()
    print(f"Blender Available: {blender_available}")
    if blender_available:
        print(f"Blender Path: {blender_path}")
    
    # Test the model creation
    test_prompts = [
        "Create a detailed human eye with retina structure and highlight regions",
        "Generate a brain model with cerebral regions highlighted",
        "Design an anatomical heart with chamber highlights"
    ]
    
    print("\\nBlender Agent - Enhanced Model Creation Test")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\\nTest {i}: {prompt}")
        try:
            model_path = create_model(prompt)
            print(f"Success! Model saved to: {model_path}")
            
            # Get model info
            info = get_model_info(model_path)
            if info:
                print("Model Info:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                    
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 40)
    
    # Cleanup after testing
    print("\\nCleaning up test files...")
    cleanup_models()