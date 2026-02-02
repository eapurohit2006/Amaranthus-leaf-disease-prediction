# Amaranthus-leaf-disease-prediction
Developed a web-based application to predict diseases in Amaranthus leaves using image-based analysis. The system helps in early disease detection and supports farmers in taking preventive actions.
## ✨ Features
- User authentication with JWT (Sign up & Login)
- Pages: Home, About, Predict
- Upload leaf image for disease prediction
- Displays predicted disease classes with probabilities
- Shows recommended precautions for detected diseases
- Pluggable machine learning model integration

## 🗂️ Folder Structure
backend/
  __init__.py
  main.py
  auth.py
  config.py
  database.py
  models.py
  model_loader.py
  precautions.py
  schemas.py
frontend/
  index.html
  about.html
  login.html
  signup.html
  predict.html
  styles.css
  shared.js
  login.js
  signup.js
  predict.js
requirements.txt

## 🧠 Model Integration
Create model/inference.py and add:
from PIL import Image
def predict(image: Image.Image):
    return [
        { 'label': 'Leaf Blight', 'probability': 0.85 },
        { 'label': 'Healthy', 'probability': 0.10 },
        { 'label': 'Leaf Spot', 'probability': 0.05 },
    ]

## ⚙️ Setup & Run (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

## 🌐 Run Frontend
python -m http.server 5500 -d frontend
Visit http://localhost:5500

## 🔑 Authentication APIs
POST /auth/signup { email, password, full_name? } -> Bearer token
POST /auth/login { email, password } -> Bearer token
Use header: Authorization: Bearer <token>

## 🌿 Predict API
POST /predict (multipart form-data: image)
Response includes predicted labels, probabilities, and precautions.

## 🚀 Tech Stack
Frontend: HTML, CSS, JavaScript
Backend: Python, FastAPI
Auth: JWT
ML: Pluggable inference module
Image Processing: Pillow (PIL)

## 📄 License
This project is intended for educational and research purposes.


