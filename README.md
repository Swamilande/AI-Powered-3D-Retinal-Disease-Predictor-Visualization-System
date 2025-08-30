# AI-Powered Retinal Disease 3D Visualizer - Backend

A Flask-based backend for visualizing retinal diseases in 3D using AI-generated Blender models.

## 🏗 Architecture


├── app.py                 # Flask entry point & API routes
├── agents/
│   ├── doctor_agent.py    # Medical analysis (mock)
│   ├── prompt_agent.py    # Gemini API integration
│   └── blender_agent.py   # Blender headless execution
├── static/models/         # Generated 3D models (.glb files)
├── uploads/              # Uploaded scan files
└── tests/                # Test files


## 🔧 Setup

### 1. Install Dependencies

bash
# Clone/download the project
pip install -r requirements.txt


### 2. Install Blender

Download and install Blender from [blender.org](https://www.blender.org/download/)

