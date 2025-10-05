#🚀 Space Station Safety Scanner
##🌌 AI-Powered Real-Time Safety Equipment Detection & Risk Analysis

The Space Station Safety Scanner is an advanced AI-driven web application that detects and analyzes critical safety equipment from images or real-time camera feeds using the DeepSeek Vision API.
It identifies objects like fire extinguishers, oxygen tanks, and first aid boxes, and provides instant safety insights, alerts, and analytics — all inside a visually immersive, animated space-themed interface.

##🧠 Features

AI-Powered Object Detection using DeepSeek Vision

Interactive and visually rich web interface

Real-Time Monitoring with continuous AI scanning

Comprehensive Safety Analytics Dashboard

Fallback Mode (works even if API fails)

Fully animated space-themed UI (stars, rockets, satellites, planets)

##🏗️ Project Structure
📁 Space-Station-Safety-Scanner/
│
├── app.py              # Flask backend with AI integration and API endpoints
├── index.html          # Frontend interface (HTML, CSS, JS)
├── render.yaml         # Render deployment configuration
└── README.md           # Project documentation

##⚙️ Installation & Setup
###1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Space-Station-Safety-Scanner.git
cd Space-Station-Safety-Scanner

###2️⃣ Install Dependencies

Make sure you have Python 3.9+ installed, then run:

pip install flask flask-cors opencv-python pillow requests

###3️⃣ Run the Flask App
python app.py


Then open your browser at:

http://127.0.0.1:5000

##🌍 API Endpoints
Endpoint	Method	Description
/	GET	Loads the main frontend interface
/api/detect	POST	Accepts base64 image data and returns AI detection results
/api/health	GET	Health check endpoint
/api/realtime/start	POST	Starts real-time detection thread
/api/realtime/stop	POST	Stops real-time detection thread
/api/realtime/analyze_frame	POST	Sends frames for continuous AI analysis
🧩 Example Usage
Example Request
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}

Example Response
{
  "success": true,
  "analysis_result": {
    "detections": [
      {
        "class_name": "Fire Extinguisher",
        "confidence": 0.94,
        "condition": "good",
        "accessibility": "accessible"
      }
    ],
    "safety_score": 85,
    "risk_assessment": "medium"
  },
  "alerts": [
    { "type": "success", "message": "GOOD: Safety systems adequate" }
  ]
}

##🛰️ Deployment on Render

This project includes a render.yaml file for automatic deployment on Render.

Steps:

Push the repository to GitHub.

Go to Render.com
.

Create a New Web Service.

Connect your GitHub repository.

Render automatically detects and deploys using render.yaml.

Once deployed, your web app will be live! 🚀

##🧠 Tech Stack

Backend: Flask (Python)

Frontend: HTML, CSS, JavaScript

AI Model: DeepSeek Vision API

Libraries: OpenCV, Pillow, Requests, Flask-CORS

Deployment: Render.com

#⚠️ Notes

Keep your DeepSeek API key secure (avoid hardcoding).

The system switches to fallback detection if API requests fail.

Works best with high-resolution, well-lit images.

#📸 Demo

Once the app is running, open your browser at:

http://127.0.0.1:5000


You can:

Upload images for AI-based safety analysis

View detected safety equipment

Check analytics and risk levels

Start live camera monitoring
