Amaranthus Leaf Disease Prediction Website

Features
- User auth: sign up, login (JWT)
- Pages: Home, About, Predict
- Upload image, get predicted class list and precautions
- Pluggable model: put your code in `model/inference.py`

Folder Structure
```
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
```

Model Integration
Create `model/inference.py` with:
```python
from PIL import Image

def predict(image: Image.Image):
    # return a list of { 'label': str, 'probability': float }
    return [
        { 'label': 'Leaf Blight', 'probability': 0.85 },
        { 'label': 'Healthy', 'probability': 0.1 },
        { 'label': 'Leaf Spot', 'probability': 0.05 },
    ]
```

Run (Windows PowerShell)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open frontend
```
python -m http.server 5500 -d frontend
```
Visit http://localhost:5500

Auth
- POST /auth/signup { email, password, full_name? } -> bearer token
- POST /auth/login { email, password } -> bearer token
- Use Authorization: Bearer <token> when calling /predict

Predict API
- POST /predict (multipart form: image)
- Response includes `predictions` with label, probability, and precautions.


