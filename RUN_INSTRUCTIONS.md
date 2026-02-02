# How to Run LeafScan Application

## Prerequisites

- **Python 3.8+** installed
- **Node.js 16+** and **npm** installed
- Model files in project root:
  - `amaranthus_resnet50v2.keras` ✅ (already present)
  - `class_indices_resnet50v2.json` ✅ (already present)

## Step 1: Setup Backend (Python FastAPI)

### Option A: Using Virtual Environment (Recommended)

```powershell
# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# If activation fails, try:
# .\.venv\Scripts\activate.bat

# Install Python dependencies
pip install -r requirements.txt

# Run the backend server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Without Virtual Environment

```powershell
# Install dependencies
pip install -r requirements.txt

# Run the backend server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will run on: **http://localhost:8000**

You can test it by visiting: http://localhost:8000/health

---

## Step 2: Setup Frontend (React + Vite)

Open a **NEW terminal window** (keep backend running) and run:

```powershell
# Navigate to frontend directory
cd frontend-react

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

The frontend will run on: **http://localhost:5173** (or another port if 5173 is busy)

---

## Step 3: Access the Application

1. Open your browser and go to: **http://localhost:5173**
2. You'll see the LeafScan login page
3. Create an account or login to start using the app

---

## Quick Start Commands Summary

### Terminal 1 (Backend):
```powershell
# Activate venv (if using)
.\.venv\Scripts\Activate.ps1

# Run backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 (Frontend):
```powershell
cd frontend-react
npm install  # First time only
npm run dev
```

---

## Troubleshooting

### Backend Issues:

1. **Port 8000 already in use:**
   ```powershell
   # Use a different port
   uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
   ```
   Then update frontend API calls in `Login.jsx`, `Signup.jsx`, and `Predict.jsx` to use port 8001

2. **Module not found errors:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Model not loading:**
   - Ensure `amaranthus_resnet50v2.keras` and `class_indices_resnet50v2.json` are in the project root
   - Check the `/health` endpoint to see if model loaded: http://localhost:8000/health

### Frontend Issues:

1. **Port 5173 already in use:**
   - Vite will automatically use the next available port
   - Check the terminal output for the actual port

2. **npm install fails:**
   ```powershell
   # Clear cache and reinstall
   npm cache clean --force
   npm install
   ```

3. **API connection errors:**
   - Ensure backend is running on port 8000
   - Check browser console for CORS errors
   - Verify API URL in frontend files matches backend port

---

## Production Build

### Build Frontend:
```powershell
cd frontend-react
npm run build
```

### Run Backend in Production:
```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

- `GET /health` - Health check
- `POST /auth/signup` - Create account
- `POST /auth/login` - Login
- `POST /predict` - Predict disease (requires authentication)

---

## Notes

- The backend must be running before using the frontend
- Authentication is required for predictions
- User data is stored in `users.db.json` (created automatically)
- Prediction history is stored in browser localStorage






