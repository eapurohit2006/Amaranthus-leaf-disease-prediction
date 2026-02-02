from __future__ import annotations
from typing import Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from fastapi import HTTPException, status
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET = "change_me_dev_secret"
ALGO = "HS256"
ACCESS_MIN = 120

USERS_FILE = Path("users.db.json")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SignupBody(BaseModel):
	username: str
	password: str
	full_name: Optional[str] = None


class LoginBody(BaseModel):
	username: str
	password: str


def _read_users() -> dict:
	if USERS_FILE.exists():
		try:
			return json.loads(USERS_FILE.read_text(encoding="utf-8"))
		except Exception:
			return {}
	return {}


def _write_users(data: dict) -> None:
	USERS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _hash(password: str) -> str:
	return pwd_context.hash(password)


def _verify(password: str, password_hash: str) -> bool:
	return pwd_context.verify(password, password_hash)


def _token_for(username: str, role: str = "user") -> str:
	exp = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_MIN)
	return jwt.encode({"sub": username, "role": role, "exp": exp}, SECRET, algorithm=ALGO)


def signup(body: SignupBody) -> dict:
	users = _read_users()
	if body.username in users:
		raise HTTPException(status_code=400, detail="Username already registered")
	
	# Signup always creates user role (not admin)
	users[body.username] = {
		"password_hash": _hash(body.password),
		"full_name": getattr(body, 'full_name', None) or "",
		"role": "user",
		"email": body.username  # Store username as email for compatibility
	}
	_write_users(users)
	token = _token_for(body.username, "user")
	return {"access_token": token, "token_type": "bearer", "role": "user"}


def login(body: LoginBody) -> dict:
	users = _read_users()
	user = users.get(body.username)
	if not user or not _verify(body.password, user.get("password_hash", "")):
		raise HTTPException(status_code=401, detail="Invalid credentials")
	
	role = user.get("role", "user")
	token = _token_for(body.username, role)
	return {"access_token": token, "token_type": "bearer", "role": role}


def require_bearer(header: Optional[str]) -> str:
	if not header or not header.lower().startswith("bearer "):
		raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
	return header.split(" ", 1)[1]


def decode_token(token: str) -> dict:
	try:
		payload = jwt.decode(token, SECRET, algorithms=[ALGO])
		username = payload.get("sub")
		role = payload.get("role", "user")
		if not username:
			raise HTTPException(status_code=401, detail="Invalid token")
		return {"username": username, "role": role}
	except JWTError:
		raise HTTPException(status_code=401, detail="Invalid token")


def get_user_role(username: str) -> str:
	"""Get user role from database"""
	users = _read_users()
	user = users.get(username)
	if not user:
		return "user"
	return user.get("role", "user")


def init_admin_user():
	"""Initialize default admin user if not exists"""
	users = _read_users()
	admin_username = "admin@example.com"
	
	if admin_username not in users:
		users[admin_username] = {
			"password_hash": _hash("Admin@123"),
			"full_name": "Admin User",
			"role": "admin",
			"email": admin_username
		}
		_write_users(users)
		print(f"✅ Default admin user created: {admin_username} / Admin@123")
	else:
		print(f"ℹ️  Admin user already exists: {admin_username}")






