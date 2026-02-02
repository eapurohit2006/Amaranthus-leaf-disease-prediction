from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import jwt, JWTError
import bcrypt
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Depends

from .config import JWT_SECRET, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from .database import get_db
from .models import User


def hash_password(password: str) -> str:
	"""Hash a password using bcrypt"""
	# Ensure password is bytes
	if isinstance(password, str):
		password = password.encode('utf-8')
	# Generate salt and hash
	salt = bcrypt.gensalt()
	hashed = bcrypt.hashpw(password, salt)
	# Return as string (bcrypt hash is always $2b$... format)
	return hashed.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
	"""Verify a password against a bcrypt hash"""
	try:
		# Check for None or empty values
		if not password or not password_hash:
			return False
		# Ensure both are bytes
		if isinstance(password, str):
			password = password.encode('utf-8')
		# Check if hash looks like bcrypt format before encoding
		if isinstance(password_hash, str):
			if not password_hash.startswith('$2'):
				return False
			password_hash = password_hash.encode('utf-8')
		else:
			if not password_hash.startswith(b'$2'):
				return False
		# Verify password
		return bcrypt.checkpw(password, password_hash)
	except Exception:
		return False


def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
	to_encode = data.copy()
	exp = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
	to_encode.update({"exp": exp})
	return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(db: Session = Depends(get_db), token: Optional[str] = None):
	if token is None:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
	try:
		payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
		email: str = payload.get("sub")
		if email is None:
			raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
	except JWTError:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
	user = db.query(User).filter(User.email == email).first()
	if not user:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
	return user


