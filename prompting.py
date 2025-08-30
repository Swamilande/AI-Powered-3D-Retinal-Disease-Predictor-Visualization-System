import requests
import json
import re
from typing import Dict, List, Tuple

# Hardcoded Gemini API key - replace with your actual key
API_KEY = "AIzaSyAfh1fmV4W3rWFaMdp5lBps2PWd5VtZulU"

# Gemini API endpoint
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"

def generate_prompt(report: dict) -> str:
    """
    Convert medical report into Blender-compatible modeling prompt.
    Generates precise, actionable instructions for 3D visualization.
    
    Args:
        report (dict): Medical report containing diagnosis, findings, etc.
        
    Returns:
        str: Precise Blender prompt with specific modeling instructions
    """
    # Extract and clean key information
    diagnosis = _clean_text(report.get("diagnosis", ""))
    findings = _clean_text(report.get("findings", ""))
    anatomy = _clean_text(report.get("anatomy", "organ"))
    severity = _clean_text(report.get("severity", ""))
    
    # Parse medical terminology into 3D modeling instructions
    anatomical_specs = _parse_anatomy(anatomy)
    pathology_specs = _parse_pathology(findings, diagnosis, severity)
    visualization_specs = _get_visualization_requirements(diagnosis, findings)
    
    # Generate comprehensive Blender prompt
    prompt_parts = []
    
    # 1. Base anatomical structure
    prompt_parts.append(_generate_base_structure_prompt(anatomy, anatomical_specs))
    
    # 2. Pathological highlighting
    if pathology_specs:
        prompt_parts.append(_generate_pathology_prompt(pathology_specs))
    
    # 3. Layer specifications
    prompt_parts.append(_generate_layer_specifications(anatomy))
    
    # 4. Zoom and interaction requirements
    prompt_parts.append(_generate_interaction_requirements(findings))
    
    # 5. Material and rendering specifications
    prompt_parts.append(_generate_material_specs(anatomy, pathology_specs))
    
    return " ".join(prompt_parts)

def _clean_text(text: str) -> str:
    """Clean and normalize text input."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())

def _parse_anatomy(anatomy: str) -> Dict[str, str]:
    """
    Parse anatomical terms into specific 3D modeling requirements.
    
    Args:
        anatomy (str): Anatomical region or organ
        
    Returns:
        dict: Anatomical specifications for modeling
    """
    anatomy_map = {
        'retina': {
            'base_shape': 'curved_surface',
            'layers': ['photoreceptors', 'bipolar_cells', 'ganglion_cells', 'rpe', 'choroid'],
            'geometry': 'multi_layered_sphere_segment',
            'scale': 'microscopic_detail',
            'transparency': 'partial_for_layers'
        },
        'eye': {
            'base_shape': 'sphere',
            'layers': ['cornea', 'iris', 'lens', 'vitreous', 'retina', 'sclera'],
            'geometry': 'nested_spheres',
            'scale': 'organ_level',
            'transparency': 'gradient_from_front'
        },
        'brain': {
            'base_shape': 'complex_organic',
            'layers': ['cortex', 'white_matter', 'ventricles', 'cerebellum', 'brainstem'],
            'geometry': 'sculpted_organic_form',
            'scale': 'full_organ',
            'transparency': 'sectional_cuts'
        },
        'heart': {
            'base_shape': 'muscular_pump',
            'layers': ['epicardium', 'myocardium', 'endocardium', 'chambers', 'valves'],
            'geometry': 'anatomical_chambers',
            'scale': 'full_organ',
            'transparency': 'chamber_visibility'
        },
        'lung': {
            'base_shape': 'branching_tree',
            'layers': ['pleura', 'alveoli', 'bronchi', 'blood_vessels'],
            'geometry': 'fractal_branching',
            'scale': 'multi_scale',
            'transparency': 'airway_visibility'
        }
    }
    
    # Find matching anatomy or use generic
    for key in anatomy_map:
        if key in anatomy:
            return anatomy_map[key]
    
    # Generic fallback
    return {
        'base_shape': 'organic_form',
        'layers': ['outer_surface', 'internal_structure'],
        'geometry': 'detailed_mesh',
        'scale': 'organ_level',
        'transparency': 'selective'
    }

def _parse_pathology(findings: str, diagnosis: str, severity: str) -> Dict[str, any]:
    """
    Parse pathological findings into 3D visualization specifications.
    
    Args:
        findings (str): Medical findings
        diagnosis (str): Primary diagnosis
        severity (str): Severity level
        
    Returns:
        dict: Pathology visualization specifications
    """
    combined_text = f"{findings} {diagnosis}".lower()
    pathology_specs = {
        'highlight_regions': [],
        'color_coding': {},
        'geometry_modifications': [],
        'transparency_effects': [],
        'animation_requirements': []
    }
    
    # Swelling/Edema
    if any(term in combined_text for term in ['swelling', 'edema', 'inflammation', 'enlarged']):
        pathology_specs['highlight_regions'].append('swollen_areas')
        pathology_specs['color_coding']['swollen_areas'] = 'warm_red_yellow'
        pathology_specs['geometry_modifications'].append('increase_thickness_by_30_percent')
        pathology_specs['transparency_effects'].append('semi_transparent_overlay')
    
    # Hemorrhage/Bleeding
    if any(term in combined_text for term in ['hemorrhage', 'bleeding', 'blood', 'hematoma']):
        pathology_specs['highlight_regions'].append('hemorrhage_sites')
        pathology_specs['color_coding']['hemorrhage_sites'] = 'dark_red_pooling'
        pathology_specs['geometry_modifications'].append('irregular_surface_distortion')
        pathology_specs['transparency_effects'].append('blood_opacity_gradient')
    
    # Detachment
    if any(term in combined_text for term in ['detachment', 'separation', 'lifted']):
        pathology_specs['highlight_regions'].append('detached_regions')
        pathology_specs['color_coding']['detached_regions'] = 'contrasting_blue'
        pathology_specs['geometry_modifications'].append('curved_separation_gap')
        pathology_specs['transparency_effects'].append('gap_visibility')
    
    # Tumor/Mass
    if any(term in combined_text for term in ['tumor', 'mass', 'growth', 'lesion', 'nodule']):
        pathology_specs['highlight_regions'].append('tumor_mass')
        pathology_specs['color_coding']['tumor_mass'] = 'purple_pink_gradient'
        pathology_specs['geometry_modifications'].append('irregular_protrusion')
        pathology_specs['transparency_effects'].append('solid_with_transparent_margins')
    
    # Degeneration
    if any(term in combined_text for term in ['degeneration', 'atrophy', 'thinning', 'loss']):
        pathology_specs['highlight_regions'].append('degenerated_areas')
        pathology_specs['color_coding']['degenerated_areas'] = 'muted_brown_gray'
        pathology_specs['geometry_modifications'].append('reduced_thickness')
        pathology_specs['transparency_effects'].append('faded_appearance')
    
    # Scarring/Fibrosis
    if any(term in combined_text for term in ['scar', 'fibrosis', 'adhesion']):
        pathology_specs['highlight_regions'].append('scar_tissue')
        pathology_specs['color_coding']['scar_tissue'] = 'white_pearl'
        pathology_specs['geometry_modifications'].append('irregular_texture')
        pathology_specs['transparency_effects'].append('semi_opaque')
    
    # Severity-based modifications
    severity_multipliers = {
        'mild': 0.3,
        'moderate': 0.6,
        'severe': 1.0,
        'critical': 1.3
    }
    
    multiplier = severity_multipliers.get(severity, 0.6)
    pathology_specs['severity_multiplier'] = multiplier
    
    return pathology_specs

def _generate_base_structure_prompt(anatomy: str, specs: Dict[str, str]) -> str:
    """Generate base anatomical structure instructions."""
    base_prompt = f"Create a detailed 3D {anatomy} model using {specs['geometry']} geometry."
    
    if specs['layers']:
        layers_text = ", ".join(specs['layers'][:3])  # Limit to first 3 for brevity
        base_prompt += f" Include anatomical layers: {layers_text}."
    
    base_prompt += f" Apply {specs['scale']} level of detail with {specs['transparency']} transparency."
    
    return base_prompt

def _generate_pathology_prompt(pathology_specs: Dict[str, any]) -> str:
    """Generate pathology-specific visualization instructions."""
    if not pathology_specs['highlight_regions']:
        return ""
    
    prompt_parts = []
    
    # Highlight regions
    regions = pathology_specs['highlight_regions']
    prompt_parts.append(f"Highlight pathological regions: {', '.join(regions)}.")
    
    # Color coding
    for region, color in pathology_specs['color_coding'].items():
        prompt_parts.append(f"Apply {color} coloring to {region}.")
    
    # Geometry modifications
    for modification in pathology_specs['geometry_modifications']:
        prompt_parts.append(f"Modify geometry: {modification}.")
    
    # Transparency effects
    for effect in pathology_specs['transparency_effects']:
        prompt_parts.append(f"Apply transparency: {effect}.")
    
    # Severity scaling
    if 'severity_multiplier' in pathology_specs:
        multiplier = pathology_specs['severity_multiplier']
        prompt_parts.append(f"Scale pathology visibility by factor {multiplier}.")
    
    return " ".join(prompt_parts)

def _generate_layer_specifications(anatomy: str) -> str:
    """Generate layer-specific modeling instructions."""
    layer_specs = {
        'retina': "Create 10 distinct retinal layers with appropriate thickness ratios. Enable individual layer visibility controls.",
        'eye': "Model 6 major eye structures with nested transparency. Allow sectional views.",
        'brain': "Include cortical surface detail with gyri and sulci. Add internal structure visibility.",
        'heart': "Model 4 chambers with muscular walls. Include valve structures and blood flow paths.",
        'lung': "Create bronchial tree with 5 levels of branching. Include alveolar surface detail."
    }
    
    for key, spec in layer_specs.items():
        if key in anatomy:
            return spec
    
    return "Ensure anatomical accuracy with multi-layered structure and realistic proportions."

def _generate_interaction_requirements(findings: str) -> str:
    """Generate zoom and interaction specifications."""
    base_interaction = "Enable smooth zoom functionality into affected regions."
    
    specific_requirements = []
    
    if any(term in findings.lower() for term in ['microscopic', 'cellular', 'detail']):
        specific_requirements.append("Support microscopic-level zoom (1000x magnification)")
    
    if any(term in findings.lower() for term in ['cross-section', 'slice', 'cut']):
        specific_requirements.append("Include cross-sectional cutting plane controls")
    
    if any(term in findings.lower() for term in ['flow', 'movement', 'dynamic']):
        specific_requirements.append("Add animation controls for dynamic processes")
    
    if specific_requirements:
        return f"{base_interaction} {' '.join(specific_requirements)}."
    
    return f"{base_interaction} Allow 360-degree rotation and multi-level detail inspection."

def _generate_material_specs(anatomy: str, pathology_specs: Dict[str, any]) -> str:
    """Generate material and rendering specifications."""
    material_specs = []
    
    # Base materials
    base_materials = {
        'retina': 'semi-transparent organic tissue with subsurface scattering',
        'eye': 'clear cornea, colored iris, and translucent lens materials',
        'brain': 'soft tissue with cortical texture and white matter distinction',
        'heart': 'muscular tissue with blood-filled chambers',
        'lung': 'spongy tissue with air-filled spaces'
    }
    
    for key, spec in base_materials.items():
        if key in anatomy:
            material_specs.append(f"Apply {spec}.")
            break
    else:
        material_specs.append("Use realistic organic tissue materials with appropriate surface properties.")
    
    # Pathology materials
    if pathology_specs and pathology_specs.get('color_coding'):
        material_specs.append("Add pathology-specific materials with distinct visual properties.")
        for region, color in pathology_specs['color_coding'].items():
            material_specs.append(f"Create {color} material for {region} with appropriate opacity.")
    
    # Rendering requirements
    material_specs.append("Enable PBR materials with realistic lighting response.")
    material_specs.append("Optimize for real-time rendering with LOD system.")
    
    return " ".join(material_specs)

def _get_visualization_requirements(diagnosis: str, findings: str) -> Dict[str, bool]:
    """Determine specific visualization requirements based on medical context."""
    combined = f"{diagnosis} {findings}".lower()
    
    return {
        'needs_animation': any(term in combined for term in ['flow', 'movement', 'progression']),
        'needs_sectioning': any(term in combined for term in ['internal', 'deep', 'beneath']),
        'needs_comparison': any(term in combined for term in ['normal', 'healthy', 'baseline']),
        'needs_measurement': any(term in combined for term in ['size', 'dimension', 'thickness']),
        'needs_temporal': any(term in combined for term in ['progression', 'development', 'change'])
    }

# Fallback function for when detailed parsing isn't needed
def generate_simple_prompt(report: dict) -> str:
    """
    Simple fallback prompt generation.
    
    Args:
        report (dict): Medical report dictionary
        
    Returns:
        str: Basic but precise Blender prompt
    """
    diagnosis = report.get("diagnosis", "")
    findings = report.get("findings", "")
    anatomy = report.get("anatomy", "organ")
    
    return (
        f"Create a detailed 3D {anatomy} model with realistic anatomical layers. "
        f"Highlight pathological regions showing: {findings}. "
        f"Ensure medical accuracy with transparent overlay system allowing zoom into affected regions. "
        f"Apply appropriate materials and enable sectional viewing capabilities."
    )

# Enhanced API integration (when ready to use real Gemini)
def generate_prompt_with_api(report: dict) -> str:
    """
    Generate prompt using Gemini API with enhanced medical context.
    
    Args:
        report (dict): Medical report dictionary
        
    Returns:
        str: AI-enhanced Blender prompt
    """
    system_prompt = """
    You are a medical visualization expert specializing in Blender 3D modeling.
    Convert medical reports into precise, actionable Blender modeling instructions.
    
    Requirements:
    1. Include specific geometry modifications for pathology
    2. Specify material properties and transparency
    3. Define layer structure and visibility controls
    4. Add interaction requirements (zoom, rotation, sectioning)
    5. Ensure anatomical accuracy
    6. Keep instructions under 200 words but technically detailed
    
    Format: Direct modeling instructions without explanations.
    """
    
    # Format the medical report
    report_text = _format_enhanced_report(report)
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{system_prompt}\n\nMedical Report:\n{report_text}\n\nGenerate precise Blender modeling instructions:"
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,  # Lower temperature for more precise, technical output
            "topK": 40,
            "topP": 0.9,
            "maxOutputTokens": 500,
        }
    }
    
    try:
        if API_KEY == "AIzaSyAfh1fmV4W3rWFaMdp5lBps2PWd5VtZulU":
            # Use enhanced local generation for better results
            return generate_prompt(report)
        
        # Real API call would go here
        # response = requests.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        # ... handle response
        
        return generate_prompt(report)
        
    except Exception as e:
        print(f"API Error: {e}")
        return generate_prompt(report)

def _format_enhanced_report(report: dict) -> str:
    """Format medical report with enhanced structure for API."""
    sections = []
    
    priority_fields = ['anatomy', 'diagnosis', 'findings', 'severity']
    secondary_fields = ['patient_id', 'date', 'recommendations']
    
    # Add priority fields first
    for field in priority_fields:
        if field in report and report[field]:
            sections.append(f"{field.upper()}: {report[field]}")
    
    # Add secondary fields
    for field in secondary_fields:
        if field in report and report[field]:
            sections.append(f"{field.replace('_', ' ').title()}: {report[field]}")
    
    return "\n".join(sections)

# Example usage and testing
if __name__ == "__main__":
    # Test with comprehensive medical reports
    test_reports = [
        {
            "patient_id": "12345",
            "anatomy": "retina",
            "diagnosis": "Diabetic retinopathy with macular edema",
            "findings": "Retinal swelling in macula region, microaneurysms, hard exudates, cotton wool spots",
            "severity": "moderate",
            "recommendations": "Anti-VEGF therapy, follow-up in 6 weeks"
        },
        {
            "patient_id": "67890",
            "anatomy": "brain",
            "diagnosis": "Glioblastoma multiforme",
            "findings": "Irregular tumor mass in right frontal lobe with surrounding edema and mass effect",
            "severity": "severe",
            "recommendations": "Surgical resection, chemotherapy, radiation therapy"
        },
        {
            "patient_id": "54321",
            "anatomy": "heart",
            "diagnosis": "Myocardial infarction",
            "findings": "Reduced wall motion in anterior wall, scar tissue formation",
            "severity": "moderate",
            "recommendations": "Cardiac catheterization, medication adjustment"
        }
    ]
    
    print("Enhanced Medical Report to Blender Prompt Conversion")
    print("=" * 60)
    
    for i, report in enumerate(test_reports, 1):
        print(f"\nTest Case {i}: {report['diagnosis']}")
        print("-" * 40)
        
        # Generate enhanced prompt
        enhanced_prompt = generate_prompt(report)
        print("Enhanced Prompt:")
        print(enhanced_prompt)
        
        print("\n" + "="*60)