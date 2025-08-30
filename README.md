# AI-Powered Retinal Disease 3D Visualizer - Backend

An AI-powered retinal scan analysis system that generates a medical report and displays a realistic 3D retina model for interactive exploration.
This demo uses a pre-built .glb 3D model as output visualization.

📂 Project Structure
project/
│── main.py              # Flask backend
│── doctor.py            # Doctor Agent (simulated analysis logic)
│── blender_agent.py     # Blender Agent (3D model generator / demo loader)
│── prompting.py         # Prompt Agent for Blender instructions
│── templates/
│   └── index.html       # Frontend (HTML + CSS + JS)
│── static/
│   └── models/
│       └── retina_demo.glb   # Pre-built 3D retina model
│── uploads/             # Uploaded scans (saved temporarily)

🚀 Features

📂 Upload OCT / MRI / retinal scans (JPG, PNG, TIFF).

🧠 Doctor Agent simulates AI-based medical diagnosis:

Diagnosis

Findings

Affected Anatomy

Severity

Recommendations

✨ Prompt Agent converts the report into Blender-compatible prompts.

🎨 Blender Agent integrates a 3D retina model (.glb).

🧊 Interactive 3D viewer with zoom, rotate, and pan using <model-viewer>.

⚙️ Installation & Setup

Clone the repo:

git clone https://github.com/your-username/retinal-analyzer-demo.git
cd retinal-analyzer-demo


Create a virtual environment:

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


Install dependencies:

pip install flask


Run the app:

python main.py


Open in browser:

http://127.0.0.1:5000/

📊 Usage

Upload a retinal scan (.jpg, .png, .tiff).

Wait for AI analysis (demo doctor agent).

View the structured medical report.

Explore the 3D retina model interactively in your browser.

🧊 3D Model

Pre-built model: static/models/retina_demo.glb.

Replace this file with your own .glb to visualize other anatomical structures.

🛠️ Tech Stack

Backend: Flask (Python)

Frontend: HTML, CSS, JavaScript

3D Rendering: <model-viewer> (Web Component, Google)

AI Agents (Simulated): Doctor Agent, Prompt Agent, Blender Agent

📌 Roadmap

✅ Demo with pre-built model

🔄 Integrate real AI analysis pipeline

🔬 Support for multiple anatomical 3D models

☁️ Deploy on cloud for real-world use
