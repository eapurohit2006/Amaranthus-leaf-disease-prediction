from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PIL import Image
import io
from contextlib import asynccontextmanager

from .config import ALLOWED_ORIGINS
from .database import Base, engine, get_db, SessionLocal
from .models import User
from .auth import hash_password, verify_password, create_access_token
from .schemas import SignupRequest, LoginRequest, TokenResponse, PredictionResponse, PredictionItem, UserResponse
from .model_loader import load_external_predict, ModelNotFound, load_predict_function
from .precautions import get_precautions


def init_admin_user():
	"""Initialize default admin user if not exists, or reset password if exists"""
	db = SessionLocal()
	try:
		admin_email = "admin@example.com"
		admin_password = "Admin@123"
		existing_admin = db.query(User).filter(User.email == admin_email).first()
		
		if not existing_admin:
			# Create new admin user
			admin_user = User(
				email=admin_email,
				password_hash=hash_password(admin_password),
				full_name="Admin User",
				role="admin"
			)
			db.add(admin_user)
			db.commit()
			print(f"Default admin user created: {admin_email} / {admin_password}")
		else:
			# Update existing admin: ensure role is set and password hash is correct
			updated = False
			if not existing_admin.role or existing_admin.role != "admin":
				existing_admin.role = "admin"
				updated = True
			# Always reset password hash to ensure it's in the correct bcrypt format
			# This fixes issues with old passlib hashes
			existing_admin.password_hash = hash_password(admin_password)
			updated = True
			
			if updated:
				db.commit()
				print(f"Admin user updated: {admin_email} / {admin_password}")
			else:
				print(f"Admin user already exists: {admin_email}")
	except Exception as e:
		print(f"Error initializing admin user: {e}")
		import traceback
		traceback.print_exc()
		db.rollback()
	finally:
		db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
	# Startup
	print("Starting application...")
	# Initialize DB
	try:
		Base.metadata.create_all(bind=engine)
		print("Database initialized")
	except Exception as e:
		print(f"Database initialization error: {e}")
	
	# Initialize admin user
	init_admin_user()
	
	# Load model predict function (external or TF fallback)
	global predict_external
	try:
		predict_external = load_predict_function()
		print("Model loaded successfully")
	except ModelNotFound as e:
		predict_external = None
		print(f"Model not found: {e}")
	except Exception as e:
		predict_external = None
		import traceback
		print(f"Error loading model: {e}")
		print(traceback.format_exc())
	
	yield
	
	# Shutdown
	print("Shutting down application...")


app = FastAPI(title="Amaranthus Leaf Disease Prediction", lifespan=lifespan)

app.add_middleware(
	CORSMiddleware,
	allow_origins=ALLOWED_ORIGINS,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Global variable for predict function
predict_external = None


@app.get("/health")
def health():
	return {
		"status": "ok", 
		"model_loaded": predict_external is not None,
		"model_type": "model/inference.py" if predict_external else "none"
	}


@app.post("/auth/signup", response_model=TokenResponse)
def signup(data: SignupRequest, db: Session = Depends(get_db)):
	try:
		# Map username to email field (database stores as email)
		existing = db.query(User).filter(User.email == data.username).first()
		if existing:
			raise HTTPException(status_code=400, detail="Username already registered")
		# Signup always creates user role (not admin)
		user = User(email=data.username, password_hash=hash_password(data.password), full_name=data.full_name or "", role="user")
		db.add(user)
		db.commit()
		db.refresh(user)
		token = create_access_token({"sub": user.email})
		return TokenResponse(access_token=token, role=user.role)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@app.post("/auth/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
	try:
		# Map username to email field (database stores as email)
		user = db.query(User).filter(User.email == data.username).first()
		if not user or not verify_password(data.password, user.password_hash):
			raise HTTPException(status_code=401, detail="Invalid credentials")
		token = create_access_token({"sub": user.email})
		# Safely get role, defaulting to "user" if column doesn't exist (for backward compatibility)
		user_role = getattr(user, 'role', 'user') or 'user'
		return TokenResponse(access_token=token, role=user_role)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


def _require_bearer(auth_header: str | None) -> str:
	if not auth_header or not auth_header.lower().startswith("bearer "):
		raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
	return auth_header.split(" ", 1)[1]


@app.get("/auth/me", response_model=UserResponse)
def get_current_user_info(authorization: str | None = Header(default=None, alias="Authorization"), db: Session = Depends(get_db)):
	"""Get current user information"""
	token = _require_bearer(authorization)
	try:
		from jose import jwt, JWTError
		from .config import JWT_SECRET, JWT_ALGORITHM
		payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
		email = payload.get("sub")
		if not email:
			raise HTTPException(status_code=401, detail="Invalid token")
	except JWTError:
		raise HTTPException(status_code=401, detail="Invalid token")
	
	user = db.query(User).filter(User.email == email).first()
	if not user:
		raise HTTPException(status_code=404, detail="User not found")
	
	return UserResponse(
		email=user.email,
		full_name=user.full_name or "",
		role=getattr(user, 'role', 'user') or 'user'
	)


@app.get("/admin/users")
def get_admin_users(authorization: str | None = Header(default=None, alias="Authorization"), db: Session = Depends(get_db)):
	"""Get all users (admin only)"""
	# Verify admin token
	token = _require_bearer(authorization)
	try:
		from jose import jwt, JWTError
		from .config import JWT_SECRET, JWT_ALGORITHM
		payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
		email = payload.get("sub")
		if not email:
			raise HTTPException(status_code=401, detail="Invalid token")
	except JWTError:
		raise HTTPException(status_code=401, detail="Invalid token")
	
	# Check if user is admin
	user = db.query(User).filter(User.email == email).first()
	if not user:
		raise HTTPException(status_code=404, detail="User not found")
	
	user_role = getattr(user, 'role', 'user') or 'user'
	if user_role != 'admin':
		raise HTTPException(status_code=403, detail="Admin access required")
	
	# Get all users
	all_users = db.query(User).all()
	return {"total": len(all_users), "users": [{"email": u.email, "full_name": u.full_name or "", "role": getattr(u, 'role', 'user') or 'user'} for u in all_users]}


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...), authorization: str | None = Header(default=None, alias="Authorization")):
	# Auth check (lightweight)
	_ = _require_bearer(authorization)

	if predict_external is None:
		raise HTTPException(status_code=503, detail="Model not found. Place model/inference.py with predict(image) function.")

	content_type = image.content_type or ""
	if not any(ct in content_type for ct in ("image/jpeg", "image/png", "image/webp")):
		raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG/PNG/WebP.")

	data = await image.read()
	try:
		img = Image.open(io.BytesIO(data)).convert("RGB")
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid image file")

	try:
		base_preds = predict_external(img)  # [{'label': str, 'probability': float}, ...]
		if not base_preds or len(base_preds) == 0:
			raise HTTPException(status_code=500, detail="Model returned no predictions")
		
		items: list[PredictionItem] = []
		for p in base_preds:
			items.append(PredictionItem(
				label=str(p.get("label", "Unknown")),
				probability=float(p.get("probability", 0.0)),
				precautions=get_precautions(str(p.get("label", "Unknown")))
			))
		return PredictionResponse(predictions=items)
	except HTTPException:
		raise
	except Exception as e:
		import traceback
		error_detail = f"Prediction failed: {str(e)}"
		error_trace = traceback.format_exc()
		print(f"Prediction error: {error_detail}")
		print(error_trace)
		# Return more detailed error for debugging
		raise HTTPException(status_code=500, detail=f"{error_detail}. Check server logs for details.")

