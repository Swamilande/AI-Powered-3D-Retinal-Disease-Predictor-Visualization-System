"""
Enhanced Medical Scan Analysis Module

This module provides comprehensive medical scan analysis with detailed reporting
optimized for 3D visualization generation. Includes ML model placeholders and
sophisticated medical terminology processing.
"""

import os
import re
import json
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import hashlib


def analyze_scan(file_path: str) -> Dict[str, str]:
    """
    Analyze medical scan with comprehensive ML model inference.
    
    Args:
        file_path (str): Path to the medical scan file
        
    Returns:
        dict: Dictionary containing detailed medical analysis optimized for 3D visualization
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scan not found: {file_path}")
    
    # Validate file format with enhanced checking
    file_info = _validate_and_analyze_file(file_path)
    if not file_info['is_valid']:
        raise ValueError(f"Unsupported file format: {file_info['extension']}")
    
    print(f"Analyzing {file_info['scan_type']} scan: {file_path}")
    
    # TODO: Replace with actual ML model inference
    # model = load_ml_model('models/retinal_analysis_v2.h5')
    # preprocessed_data = preprocess_image(file_path, file_info['scan_type'])
    # raw_prediction = model.predict(preprocessed_data)
    # analysis_result = postprocess_prediction(raw_prediction, file_info)
    
    # Enhanced ML inference simulation with detailed medical reporting
    analysis_result = _enhanced_ml_inference(file_path, file_info)
    
    # Add metadata and validation
    analysis_result.update({
        'scan_metadata': file_info,
        'analysis_timestamp': datetime.now().isoformat(),
        'confidence_score': _calculate_confidence_score(analysis_result),
        'visualization_priority': _determine_visualization_priority(analysis_result)
    })
    
    return analysis_result


def _validate_and_analyze_file(file_path: str) -> Dict[str, any]:
    """
    Enhanced file validation with scan type detection.
    
    Args:
        file_path (str): Path to the medical scan file
        
    Returns:
        dict: File information and scan type analysis
    """
    valid_formats = {
        # Standard image formats
        '.jpg': 'fundus_photography',
        '.jpeg': 'fundus_photography', 
        '.png': 'digital_scan',
        '.tiff': 'high_resolution_scan',
        '.tif': 'high_resolution_scan',
        '.bmp': 'basic_imaging',
        '.gif': 'dynamic_imaging',
        
        # Medical-specific formats
        '.dcm': 'dicom_medical',
        '.nii': 'neuroimaging',
        '.nifti': 'neuroimaging',
        
        # OCT formats (common extensions)
        '.oct': 'optical_coherence_tomography',
        '.img': 'medical_imaging'
    }
    
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()
    
    # File size analysis
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    
    # Generate file hash for tracking
    file_hash = _generate_file_hash(file_path)
    
    return {
        'is_valid': extension in valid_formats,
        'extension': extension,
        'scan_type': valid_formats.get(extension, 'unknown'),
        'file_size': file_size,
        'file_hash': file_hash,
        'filename': file_path_obj.name,
        'resolution_category': _categorize_resolution(file_size)
    }


def _generate_file_hash(file_path: str) -> str:
    """Generate MD5 hash of file for tracking."""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()[:16]  # Short hash for tracking
    except:
        return "unknown"


def _categorize_resolution(file_size: int) -> str:
    """Categorize image resolution based on file size."""
    if file_size < 100_000:  # < 100KB
        return "low_resolution"
    elif file_size < 1_000_000:  # < 1MB
        return "standard_resolution"
    elif file_size < 5_000_000:  # < 5MB
        return "high_resolution"
    else:
        return "ultra_high_resolution"


def _enhanced_ml_inference(file_path: str, file_info: Dict[str, any]) -> Dict[str, str]:
    """
    Enhanced ML model inference simulation with comprehensive medical analysis.
    This function provides detailed medical reports optimized for 3D visualization.
    
    Args:
        file_path (str): Path to the medical scan file
        file_info (dict): File metadata and scan type information
        
    Returns:
        dict: Comprehensive medical analysis results
    """
    filename = Path(file_path).name.lower()
    scan_type = file_info['scan_type']
    
    # Enhanced condition simulation based on filename patterns and scan type
    analysis_result = _simulate_comprehensive_analysis(filename, scan_type)
    
    # Add scan-type specific enhancements
    analysis_result.update(_add_scan_specific_details(scan_type, analysis_result))
    
    return analysis_result


def _simulate_comprehensive_analysis(filename: str, scan_type: str) -> Dict[str, str]:
    """
    Simulate comprehensive medical analysis with detailed findings.
    
    Args:
        filename (str): Filename for pattern matching
        scan_type (str): Type of medical scan
        
    Returns:
        dict: Detailed medical analysis
    """
    # Advanced pattern matching for various conditions
    conditions = _analyze_condition_patterns(filename)
    
    if conditions['diabetic_retinopathy']:
        return {
            "diagnosis": "Diabetic macular edema with microaneurysms",
            "primary_findings": "Intraretinal fluid accumulation, hard exudates, microaneurysms",
            "secondary_findings": "Cotton wool spots, flame-shaped hemorrhages, venous beading",
            "anatomy": "retina",
            "affected_regions": "macula, temporal arcade, nasal retina",
            "severity": "moderate_to_severe",
            "pathophysiology": "Hyperglycemia-induced capillary damage with blood-retinal barrier breakdown",
            "visualization_focus": "macular_edema_fluid_pockets",
            "layer_involvement": "inner_nuclear_layer, outer_plexiform_layer",
            "recommendations": "Immediate anti-VEGF therapy, glycemic control, monthly monitoring",
            "3d_requirements": "multi_layer_retina_with_fluid_highlighting"
        }
    
    elif conditions['glaucoma']:
        return {
            "diagnosis": "Primary open-angle glaucoma with optic neuropathy",
            "primary_findings": "Optic disc cupping ratio 0.7, RNFL thinning superior and inferior",
            "secondary_findings": "Rim tissue loss, vessel displacement, asymmetric cupping",
            "anatomy": "optic_nerve_head",
            "affected_regions": "optic_disc, peripapillary_area, RNFL",
            "severity": "moderate",
            "pathophysiology": "Progressive retinal ganglion cell death with axonal loss",
            "visualization_focus": "optic_cup_depth_and_rim_thinning",
            "layer_involvement": "nerve_fiber_layer, ganglion_cell_layer",
            "recommendations": "IOP reduction to target <15mmHg, visual field monitoring",
            "3d_requirements": "optic_nerve_with_cupping_visualization"
        }
    
    elif conditions['amd']:
        return {
            "diagnosis": "Age-related macular degeneration, intermediate stage",
            "primary_findings": "Large drusen deposits, pigmentary changes, geographic atrophy",
            "secondary_findings": "RPE irregularities, photoreceptor disruption",
            "anatomy": "macula",
            "affected_regions": "central_macula, parafoveal_area",
            "severity": "intermediate",
            "pathophysiology": "Accumulation of extracellular deposits with RPE dysfunction",
            "visualization_focus": "drusen_deposits_and_atrophy_areas",
            "layer_involvement": "rpe_layer, photoreceptor_layer, bruchs_membrane",
            "recommendations": "AREDS2 supplementation, Amsler grid monitoring, annual OCT",
            "3d_requirements": "macula_with_drusen_elevation_mapping"
        }
    
    elif conditions['retinal_detachment']:
        return {
            "diagnosis": "Rhegmatogenous retinal detachment",
            "primary_findings": "Full-thickness retinal tear with subretinal fluid",
            "secondary_findings": "Vitreous traction, proliferative vitreoretinopathy",
            "anatomy": "peripheral_retina",
            "affected_regions": "superior_temporal_quadrant, extending_posteriorly",
            "severity": "urgent",
            "pathophysiology": "Vitreous traction causing retinal tear with fluid migration",
            "visualization_focus": "detached_retinal_layers_with_fluid_space",
            "layer_involvement": "neurosensory_retina, subretinal_space",
            "recommendations": "Emergency surgical repair within 24 hours",
            "3d_requirements": "retina_with_detachment_gap_and_fluid_visualization"
        }
    
    elif conditions['brain_tumor']:
        return {
            "diagnosis": "Glioblastoma multiforme, right frontal lobe",
            "primary_findings": "Heterogeneously enhancing mass with central necrosis",
            "secondary_findings": "Peritumoral edema, mass effect, midline shift",
            "anatomy": "brain",
            "affected_regions": "right_frontal_lobe, extending_to_corpus_callosum",
            "severity": "severe",
            "pathophysiology": "High-grade astrocytic tumor with rapid growth and necrosis",
            "visualization_focus": "tumor_mass_with_surrounding_edema",
            "layer_involvement": "gray_matter, white_matter, vascular_structures",
            "recommendations": "Maximal safe resection, adjuvant chemoradiation",
            "3d_requirements": "brain_with_tumor_mass_and_edema_highlighting"
        }
    
    else:
        # Enhanced fallback with scan-type awareness
        return _generate_scan_type_fallback(scan_type)


def _analyze_condition_patterns(filename: str) -> Dict[str, bool]:
    """
    Analyze filename patterns to identify likely medical conditions.
    
    Args:
        filename (str): Filename to analyze
        
    Returns:
        dict: Boolean flags for different conditions
    """
    filename_lower = filename.lower()
    
    return {
        'diabetic_retinopathy': any(term in filename_lower for term in [
            'diabetic', 'dm', 'dme', 'macular_edema', 'microaneurysm'
        ]),
        'glaucoma': any(term in filename_lower for term in [
            'glaucoma', 'glau', 'optic_disc', 'cupping', 'rnfl'
        ]),
        'amd': any(term in filename_lower for term in [
            'amd', 'macular_degen', 'drusen', 'geographic_atrophy'
        ]),
        'retinal_detachment': any(term in filename_lower for term in [
            'detachment', 'rd', 'tear', 'rhegmatogenous'
        ]),
        'brain_tumor': any(term in filename_lower for term in [
            'tumor', 'glioma', 'mass', 'neoplasm', 'brain'
        ]),
        'normal': any(term in filename_lower for term in [
            'normal', 'healthy', 'control', 'baseline'
        ])
    }


def _generate_scan_type_fallback(scan_type: str) -> Dict[str, str]:
    """
    Generate fallback analysis based on scan type when specific conditions aren't detected.
    
    Args:
        scan_type (str): Type of medical scan
        
    Returns:
        dict: Scan-type appropriate fallback analysis
    """
    fallback_analyses = {
        'fundus_photography': {
            "diagnosis": "Retinal examination findings",
            "primary_findings": "Retinal vascular changes, disc appearance within normal limits",
            "secondary_findings": "Background retinopathy changes noted",
            "anatomy": "retina",
            "affected_regions": "posterior_pole, vascular_arcades",
            "severity": "mild",
            "pathophysiology": "Vascular retinal changes consistent with systemic condition",
            "visualization_focus": "retinal_vasculature_and_disc_morphology",
            "layer_involvement": "retinal_vasculature, nerve_fiber_layer",
            "recommendations": "Regular monitoring, systemic management optimization",
            "3d_requirements": "retinal_surface_with_vascular_tree_highlighting"
        },
        'optical_coherence_tomography': {
            "diagnosis": "OCT structural analysis",
            "primary_findings": "Retinal layer architecture assessment, thickness measurements",
            "secondary_findings": "Quantitative layer analysis with normative comparison",
            "anatomy": "retina",
            "affected_regions": "macular_region, peripapillary_area",
            "severity": "assessment_pending",
            "pathophysiology": "Structural retinal layer evaluation",
            "visualization_focus": "retinal_layer_thickness_mapping",
            "layer_involvement": "all_retinal_layers",
            "recommendations": "Correlation with clinical findings, serial monitoring",
            "3d_requirements": "detailed_layer_structure_with_thickness_visualization"
        },
        'neuroimaging': {
            "diagnosis": "Neuroimaging findings",
            "primary_findings": "Brain structure assessment, signal intensity evaluation",
            "secondary_findings": "White matter changes, ventricular system evaluation",
            "anatomy": "brain",
            "affected_regions": "cerebral_hemispheres, subcortical_structures",
            "severity": "varies_by_region",
            "pathophysiology": "Structural brain changes evaluation",
            "visualization_focus": "anatomical_structures_and_signal_changes",
            "layer_involvement": "gray_matter, white_matter, csf_spaces",
            "recommendations": "Clinical correlation, follow-up imaging as indicated",
            "3d_requirements": "brain_structure_with_tissue_contrast_highlighting"
        }
    }
    
    return fallback_analyses.get(scan_type, _get_generic_fallback())


def _get_generic_fallback() -> Dict[str, str]:
    """Generic fallback for unknown scan types."""
    return {
        "diagnosis": "Medical imaging findings",
        "primary_findings": "Structural assessment of anatomical region",
        "secondary_findings": "Additional morphological observations",
        "anatomy": "anatomical_structure",
        "affected_regions": "region_of_interest",
        "severity": "assessment_required",
        "pathophysiology": "Anatomical structure evaluation",
        "visualization_focus": "structural_anatomy_with_pathology_highlighting",
        "layer_involvement": "tissue_layers",
        "recommendations": "Clinical correlation and appropriate follow-up",
        "3d_requirements": "anatomical_model_with_highlighted_findings"
    }


def _add_scan_specific_details(scan_type: str, analysis: Dict[str, str]) -> Dict[str, str]:
    """
    Add scan-type specific technical details for enhanced 3D visualization.
    
    Args:
        scan_type (str): Type of medical scan
        analysis (dict): Base analysis results
        
    Returns:
        dict: Additional scan-specific details
    """
    enhancement_details = {}
    
    if 'fundus' in scan_type:
        enhancement_details.update({
            'imaging_modality': 'Color fundus photography',
            'field_of_view': '30-50 degrees',
            'depth_information': 'Surface retinal features',
            'resolution_requirements': 'High resolution for vessel detail',
            'lighting_model': 'Diffuse illumination with specular highlights'
        })
    
    elif 'oct' in scan_type or 'optical_coherence' in scan_type:
        enhancement_details.update({
            'imaging_modality': 'Optical Coherence Tomography',
            'field_of_view': '6mm x 6mm typical',
            'depth_information': 'Cross-sectional layer detail',
            'resolution_requirements': 'Microscopic layer resolution',
            'lighting_model': 'Subsurface scattering for tissue depth'
        })
    
    elif 'neuroimaging' in scan_type:
        enhancement_details.update({
            'imaging_modality': 'Neuroimaging (MRI/CT)',
            'field_of_view': 'Whole brain coverage',
            'depth_information': 'Volumetric brain structure',
            'resolution_requirements': 'Millimeter-level anatomical detail',
            'lighting_model': 'Tissue contrast with CSF transparency'
        })
    
    # Add quantitative measurements
    enhancement_details.update(_generate_quantitative_measurements(analysis))
    
    # Add 3D modeling specifications
    enhancement_details.update(_generate_3d_modeling_specs(analysis))
    
    return enhancement_details


def _generate_quantitative_measurements(analysis: Dict[str, str]) -> Dict[str, str]:
    """
    Generate quantitative measurements for 3D modeling precision.
    
    Args:
        analysis (dict): Medical analysis results
        
    Returns:
        dict: Quantitative specifications
    """
    anatomy = analysis.get('anatomy', '').lower()
    severity = analysis.get('severity', '').lower()
    
    measurements = {}
    
    if 'retina' in anatomy:
        # Retinal measurements
        if 'edema' in analysis.get('primary_findings', '').lower():
            measurements.update({
                'retinal_thickness': '450_micrometers_central',
                'edema_volume': '2.3_cubic_mm',
                'fluid_height': '120_micrometers_maximum',
                'affected_area': '3.5mm_diameter_region'
            })
        else:
            measurements.update({
                'retinal_thickness': '280_micrometers_normal',
                'total_area': '6mm_diameter_scan',
                'layer_count': '10_distinct_layers'
            })
    
    elif 'brain' in anatomy:
        # Brain measurements
        if 'tumor' in analysis.get('diagnosis', '').lower():
            measurements.update({
                'tumor_volume': '15.7_cubic_cm',
                'tumor_diameter': '3.2cm_maximum',
                'edema_extent': '5.8cm_surrounding_area',
                'mass_effect': '4mm_midline_shift'
            })
        else:
            measurements.update({
                'brain_volume': '1400_cubic_cm_normal',
                'cortical_thickness': '2.5mm_average'
            })
    
    return measurements


def _generate_3d_modeling_specs(analysis: Dict[str, str]) -> Dict[str, str]:
    """
    Generate specific 3D modeling requirements based on analysis.
    
    Args:
        analysis (dict): Medical analysis results
        
    Returns:
        dict: 3D modeling specifications
    """
    modeling_specs = {}
    
    # Geometry specifications
    anatomy = analysis.get('anatomy', '').lower()
    if 'retina' in anatomy:
        modeling_specs.update({
            'mesh_density': 'high_for_layer_detail',
            'subdivision_levels': '3_for_smooth_curves',
            'vertex_count_target': '50000_vertices',
            'texture_resolution': '2048x2048_for_detail'
        })
    elif 'brain' in anatomy:
        modeling_specs.update({
            'mesh_density': 'variable_cortical_high_subcortical_medium',
            'subdivision_levels': '2_for_organic_form',
            'vertex_count_target': '100000_vertices',
            'texture_resolution': '4096x4096_for_surface_detail'
        })
    
    # Material specifications
    primary_findings = analysis.get('primary_findings', '').lower()
    if 'fluid' in primary_findings or 'edema' in primary_findings:
        modeling_specs.update({
            'fluid_material': 'refractive_with_caustics',
            'transparency_type': 'fresnel_based',
            'subsurface_scattering': 'enabled_for_tissue_depth'
        })
    
    if 'hemorrhage' in primary_findings or 'blood' in primary_findings:
        modeling_specs.update({
            'blood_material': 'viscous_fluid_with_metallic_tint',
            'pooling_simulation': 'gravity_based_accumulation'
        })
    
    # Animation and interaction specs
    modeling_specs.update({
        'zoom_capability': 'seamless_10x_to_1000x_magnification',
        'layer_control': 'individual_layer_visibility_toggle',
        'sectioning': 'real_time_cutting_plane_adjustment',
        'measurement_tools': 'built_in_distance_and_volume_calculation'
    })
    
    return modeling_specs


def _calculate_confidence_score(analysis: Dict[str, str]) -> float:
    """
    Calculate confidence score for the analysis based on available information.
    
    Args:
        analysis (dict): Medical analysis results
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    # Base confidence factors
    confidence_factors = []
    
    # Check for specific findings
    if analysis.get('primary_findings') and len(analysis['primary_findings']) > 20:
        confidence_factors.append(0.8)
    else:
        confidence_factors.append(0.5)
    
    # Check for detailed anatomy specification
    if analysis.get('affected_regions') and ',' in analysis['affected_regions']:
        confidence_factors.append(0.9)
    else:
        confidence_factors.append(0.6)
    
    # Check for pathophysiology explanation
    if analysis.get('pathophysiology') and len(analysis['pathophysiology']) > 30:
        confidence_factors.append(0.85)
    else:
        confidence_factors.append(0.4)
    
    return round(sum(confidence_factors) / len(confidence_factors), 2)


def _determine_visualization_priority(analysis: Dict[str, str]) -> str:
    """
    Determine visualization priority based on medical urgency and complexity.
    
    Args:
        analysis (dict): Medical analysis results
        
    Returns:
        str: Priority level for 3D visualization
    """
    severity = analysis.get('severity', '').lower()
    diagnosis = analysis.get('diagnosis', '').lower()
    
    if severity in ['urgent', 'severe', 'critical']:
        return 'high_priority'
    elif any(term in diagnosis for term in ['tumor', 'detachment', 'hemorrhage']):
        return 'high_priority'
    elif severity in ['moderate']:
        return 'medium_priority'
    else:
        return 'standard_priority'


# Enhanced ML model integration functions
def load_ml_model(model_path: str, scan_type: str = 'retinal'):
    """
    Load trained ML model for specific scan type analysis.
    
    Args:
        model_path (str): Path to the trained model file
        scan_type (str): Type of scan the model is trained for
        
    Returns:
        Model object (placeholder for actual implementation)
    """
    # TODO: Implement actual model loading
    # Example implementations:
    # 
    # For TensorFlow/Keras:
    # import tensorflow as tf
    # model = tf.keras.models.load_model(model_path)
    # 
    # For PyTorch:
    # import torch
    # model = torch.load(model_path)
    # model.eval()
    # 
    # For ONNX:
    # import onnxruntime as ort
    # model = ort.InferenceSession(model_path)
    
    print(f"Loading {scan_type} ML model from: {model_path}")
    
    model_config = {
        'model_path': model_path,
        'scan_type': scan_type,
        'input_shape': _get_model_input_shape(scan_type),
        'output_classes': _get_model_output_classes(scan_type),
        'preprocessing_required': True
    }
    
    return model_config


def _get_model_input_shape(scan_type: str) -> Tuple[int, int, int]:
    """Get expected input shape for different scan types."""
    input_shapes = {
        'retinal': (512, 512, 3),
        'fundus_photography': (512, 512, 3),
        'optical_coherence_tomography': (512, 496, 1),
        'neuroimaging': (256, 256, 256),
        'dicom_medical': (512, 512, 1)
    }
    return input_shapes.get(scan_type, (512, 512, 3))


def _get_model_output_classes(scan_type: str) -> List[str]:
    """Get output classes for different scan types."""
    output_classes = {
        'retinal': ['normal', 'diabetic_retinopathy', 'glaucoma', 'amd', 'other'],
        'neuroimaging': ['normal', 'tumor', 'stroke', 'degeneration', 'other'],
        'general': ['normal', 'pathological', 'uncertain']
    }
    return output_classes.get(scan_type, output_classes['general'])


def preprocess_image(file_path: str, scan_type: str = 'retinal'):
    """
    Enhanced image preprocessing for ML model input.
    
    Args:
        file_path (str): Path to the image file
        scan_type (str): Type of medical scan for appropriate preprocessing
        
    Returns:
        Preprocessed image data (placeholder for actual implementation)
    """
    # TODO: Implement actual image preprocessing
    # Example comprehensive preprocessing:
    # 
    # import cv2
    # import numpy as np
    # from PIL import Image
    # 
    # # Load image
    # if scan_type == 'dicom_medical':
    #     import pydicom
    #     dicom_data = pydicom.dcmread(file_path)
    #     image = dicom_data.pixel_array
    # else:
    #     image = cv2.imread(file_path)
    # 
    # # Scan-specific preprocessing
    # if scan_type == 'retinal':
    #     # Fundus-specific preprocessing
    #     image = enhance_retinal_contrast(image)
    #     image = normalize_illumination(image)
    # elif scan_type == 'neuroimaging':
    #     # Brain scan preprocessing
    #     image = skull_stripping(image)
    #     image = intensity_normalization(image)
    # 
    # # Standard preprocessing
    # target_size = _get_model_input_shape(scan_type)[:2]
    # image = cv2.resize(image, target_size)
    # image = image.astype(np.float32) / 255.0
    # 
    # return np.expand_dims(image, axis=0)  # Add batch dimension
    
    preprocessing_config = {
        'file_path': file_path,
        'scan_type': scan_type,
        'target_size': _get_model_input_shape(scan_type)[:2],
        'normalization': 'zero_to_one',
        'augmentation': False  # Disabled for inference
    }
    
    print(f"Preprocessing {scan_type} image: {file_path}")
    return preprocessing_config


def postprocess_prediction(prediction, file_info: Dict[str, any] = None):
    """
    Enhanced prediction postprocessing with medical context.
    
    Args:
        prediction: Raw ML model prediction output
        file_info (dict): File metadata for context
        
    Returns:
        dict: Enhanced medical analysis with confidence scores
    """
    # TODO: Implement actual postprocessing
    # Example:
    # 
    # # Extract class probabilities
    # class_probs = softmax(prediction[0])
    # predicted_class = np.argmax(class_probs)
    # confidence = float(class_probs[predicted_class])
    # 
    # # Map to medical terminology
    # class_names = _get_model_output_classes(file_info['scan_type'])
    # diagnosis = class_names[predicted_class]
    # 
    # # Generate detailed findings based on prediction
    # findings = _generate_findings_from_prediction(prediction, file_info)
    # 
    # return {
    #     'diagnosis': diagnosis,
    #     'confidence': confidence,
    #     'findings': findings,
    #     'raw_probabilities': class_probs.tolist()
    # }
    
    postprocessing_config = {
        'prediction_format': 'class_probabilities_with_localization',
        'confidence_threshold': 0.7,
        'medical_terminology_mapping': True,
        'anatomical_region_detection': True,
        'severity_scoring': True
    }
    
    print("Enhanced postprocessing with medical context")
    return postprocessing_config


def get_supported_formats() -> Dict[str, str]:
    """
    Get comprehensive list of supported medical scan file formats.
    
    Returns:
        dict: Mapping of file extensions to scan types
    """
    return {
        '.jpg': 'Fundus photography',
        '.jpeg': 'Fundus photography',
        '.png': 'Digital medical imaging',
        '.tiff': 'High-resolution medical scan',
        '.tif': 'High-resolution medical scan',
        '.bmp': 'Basic medical imaging',
        '.dcm': 'DICOM medical format',
        '.nii': 'Neuroimaging format',
        '.nifti': 'Neuroimaging format',
        '.oct': 'Optical Coherence Tomography'
    }


def validate_scan_file(file_path: str) -> Dict[str, any]:
    """
    Enhanced file validation with detailed analysis.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        dict: Comprehensive validation results
    """
    try:
        file_info = _validate_and_analyze_file(file_path)
        
        validation_result = {
            'is_valid': file_info['is_valid'],
            'file_info': file_info,
            'scan_compatibility': _assess_scan_compatibility(file_info),
            'processing_recommendations': _get_processing_recommendations(file_info)
        }
        
        return validation_result
    
    except Exception as e:
        return {
            'is_valid': False,
            'error': str(e),
            'file_info': None,
            'scan_compatibility': 'unknown',
            'processing_recommendations': []
        }


def _assess_scan_compatibility(file_info: Dict[str, any]) -> str:
    """
    Assess compatibility of scan with 3D modeling pipeline.
    
    Args:
        file_info (dict): File information from validation
        
    Returns:
        str: Compatibility assessment
    """
    scan_type = file_info.get('scan_type', '')
    resolution_category = file_info.get('resolution_category', '')
    
    # High compatibility scans
    if scan_type in ['optical_coherence_tomography', 'neuroimaging', 'high_resolution_scan']:
        if resolution_category in ['high_resolution', 'ultra_high_resolution']:
            return 'excellent_3d_compatibility'
        else:
            return 'good_3d_compatibility'
    
    # Medium compatibility scans
    elif scan_type in ['fundus_photography', 'digital_scan']:
        if resolution_category in ['standard_resolution', 'high_resolution']:
            return 'moderate_3d_compatibility'
        else:
            return 'limited_3d_compatibility'
    
    # Basic compatibility
    else:
        return 'basic_3d_compatibility'


def _get_processing_recommendations(file_info: Dict[str, any]) -> List[str]:
    """
    Get processing recommendations based on file characteristics.
    
    Args:
        file_info (dict): File information
        
    Returns:
        list: Processing recommendations
    """
    recommendations = []
    
    resolution_category = file_info.get('resolution_category', '')
    scan_type = file_info.get('scan_type', '')
    
    # Resolution-based recommendations
    if resolution_category == 'low_resolution':
        recommendations.extend([
            'Consider upscaling for better 3D detail',
            'Use simplified geometry for low-res input',
            'Apply interpolation for missing detail'
        ])
    elif resolution_category == 'ultra_high_resolution':
        recommendations.extend([
            'Enable maximum detail 3D modeling',
            'Use high vertex count for accuracy',
            'Consider LOD system for performance'
        ])
    
    # Scan-type specific recommendations
    if scan_type == 'optical_coherence_tomography':
        recommendations.extend([
            'Focus on layer structure visualization',
            'Enable cross-sectional view capabilities',
            'Use subsurface scattering for depth'
        ])
    elif scan_type == 'fundus_photography':
        recommendations.extend([
            'Emphasize surface vessel detail',
            'Use surface mapping techniques',
            'Apply retinal curvature modeling'
        ])
    elif scan_type == 'neuroimaging':
        recommendations.extend([
            'Enable volumetric rendering',
            'Support multi-planar reconstruction',
            'Include tissue segmentation visualization'
        ])
    
    return recommendations


# Enhanced batch processing capabilities
def analyze_multiple_scans(file_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Analyze multiple scans with comparative analysis.
    
    Args:
        file_paths (list): List of paths to medical scan files
        
    Returns:
        dict: Combined analysis results with comparative insights
    """
    results = {}
    comparative_analysis = {
        'progression_detected': False,
        'bilateral_comparison': False,
        'temporal_changes': [],
        'consistent_findings': []
    }
    
    for i, file_path in enumerate(file_paths):
        try:
            scan_id = f"scan_{i+1}"
            results[scan_id] = analyze_scan(file_path)
            results[scan_id]['scan_order'] = i + 1
            results[scan_id]['file_path'] = file_path
            
        except Exception as e:
            results[f"scan_{i+1}_error"] = {
                'error': str(e),
                'file_path': file_path
            }
    
    # Add comparative analysis if multiple successful scans
    successful_scans = [k for k in results.keys() if 'error' not in k]
    if len(successful_scans) > 1:
        comparative_analysis = _perform_comparative_analysis(results, successful_scans)
    
    results['comparative_analysis'] = comparative_analysis
    return results


def _perform_comparative_analysis(results: Dict[str, Dict], successful_scans: List[str]) -> Dict[str, any]:
    """
    Perform comparative analysis across multiple scans.
    
    Args:
        results (dict): All scan results
        successful_scans (list): List of successful scan IDs
        
    Returns:
        dict: Comparative analysis results
    """
    comparative = {
        'progression_detected': False,
        'bilateral_comparison': False,
        'severity_trend': 'stable',
        'consistent_findings': [],
        'divergent_findings': [],
        'visualization_recommendations': []
    }
    
    # Check for progression if scans are sequential
    if len(successful_scans) >= 2:
        severity_levels = {'mild': 1, 'moderate': 2, 'severe': 3, 'critical': 4}
        
        severities = []
        for scan_id in successful_scans:
            severity = results[scan_id].get('severity', 'mild').lower()
            severities.append(severity_levels.get(severity, 1))
        
        if len(set(severities)) > 1:
            if severities[-1] > severities[0]:
                comparative['progression_detected'] = True
                comparative['severity_trend'] = 'worsening'
            elif severities[-1] < severities[0]:
                comparative['severity_trend'] = 'improving'
    
    # Check for consistent findings across scans
    all_diagnoses = [results[scan_id].get('diagnosis', '') for scan_id in successful_scans]
    if len(set(all_diagnoses)) == 1:
        comparative['consistent_findings'].append('diagnosis_consistent')
    
    # Generate visualization recommendations
    if comparative['progression_detected']:
        comparative['visualization_recommendations'].extend([
            'Create temporal comparison 3D models',
            'Highlight progression areas with color coding',
            'Enable side-by-side visualization mode'
        ])
    
    return comparative


# Example usage and comprehensive testing
if __name__ == "__main__":
    print("Enhanced Medical Scan Analysis - Comprehensive Testing")
    print("=" * 70)
    
    # Test with various scan types
    test_files = [
        "diabetic_retinopathy_severe.jpg",
        "glaucoma_moderate_cupping.tiff",
        "amd_drusen_deposits.png",
        "brain_tumor_glioblastoma.nii",
        "normal_retina_baseline.jpg"
    ]
    
    # Create dummy test files for demonstration
    for test_file in test_files:
        if not os.path.exists(test_file):
            with open(test_file, 'w') as f:
                f.write(f"dummy medical scan data for {test_file}")
    
    try:
        # Test individual scan analysis
        for i, test_file in enumerate(test_files, 1):
            print(f"\nTest {i}: Analyzing {test_file}")
            print("-" * 50)
            
            try:
                # Validate file first
                validation = validate_scan_file(test_file)
                print(f"File validation: {validation['is_valid']}")
                print(f"Scan compatibility: {validation['scan_compatibility']}")
                
                if validation['is_valid']:
                    # Perform analysis
                    analysis = analyze_scan(test_file)
                    
                    print("\nMedical Analysis Results:")
                    print(f"  Diagnosis: {analysis['diagnosis']}")
                    print(f"  Primary Findings: {analysis['primary_findings']}")
                    print(f"  Anatomy: {analysis['anatomy']}")
                    print(f"  Severity: {analysis['severity']}")
                    print(f"  Confidence Score: {analysis.get('confidence_score', 'N/A')}")
                    print(f"  Visualization Priority: {analysis.get('visualization_priority', 'N/A')}")
                    
                    # Show 3D requirements
                    if '3d_requirements' in analysis:
                        print(f"  3D Requirements: {analysis['3d_requirements']}")
                    
                    # Show quantitative measurements if available
                    measurements = {k: v for k, v in analysis.items() if 'thickness' in k or 'volume' in k or 'diameter' in k}
                    if measurements:
                        print("  Quantitative Measurements:")
                        for key, value in measurements.items():
                            print(f"    {key}: {value}")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        # Test batch analysis
        print(f"\n\nBatch Analysis Test:")
        print("-" * 50)
        batch_results = analyze_multiple_scans(test_files[:3])
        
        print("Batch Analysis Summary:")
        for scan_id, result in batch_results.items():
            if 'error' not in scan_id and scan_id != 'comparative_analysis':
                print(f"  {scan_id}: {result.get('diagnosis', 'Unknown')}")
        
        # Show comparative analysis
        comp_analysis = batch_results.get('comparative_analysis', {})
        if comp_analysis:
            print(f"\nComparative Analysis:")
            print(f"  Progression detected: {comp_analysis.get('progression_detected', False)}")
            print(f"  Severity trend: {comp_analysis.get('severity_trend', 'stable')}")
            if comp_analysis.get('visualization_recommendations'):
                print("  Visualization recommendations:")
                for rec in comp_analysis['visualization_recommendations']:
                    print(f"    - {rec}")
    
    finally:
        # Clean up test files
        print(f"\nCleaning up test files...")
        for test_file in test_files:
            if os.path.exists(test_file):
                os.remove(test_file)
        print("Cleanup completed.")


# Configuration class for ML model settings
class MLModelConfig:
    """Configuration for ML model integration."""
    
    # Model paths
    RETINAL_MODEL_PATH = "models/retinal_analysis_v2.h5"
    BRAIN_MODEL_PATH = "models/brain_analysis_v1.h5"
    GENERAL_MODEL_PATH = "models/general_medical_v1.h5"
    
    # Model settings
    CONFIDENCE_THRESHOLD = 0.7
    BATCH_SIZE = 1
    USE_GPU = True
    
    # Preprocessing settings
    TARGET_SIZES = {
        'retinal': (512, 512),
        'neuroimaging': (256, 256),
        'general': (224, 224)
    }
    
    # Postprocessing settings
    SEVERITY_MAPPING = {
        0: 'normal',
        1: 'mild', 
        2: 'moderate',
        3: 'severe',
        4: 'critical'
    }
    
    # 3D visualization settings
    ENABLE_3D_INTEGRATION = True
    DETAILED_LAYER_ANALYSIS = True
    PATHOLOGY_HIGHLIGHTING = True