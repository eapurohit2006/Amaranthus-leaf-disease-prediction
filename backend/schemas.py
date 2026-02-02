from pydantic import BaseModel
from typing import List


class SignupRequest(BaseModel):
	username: str
	password: str
	full_name: str | None = None


class LoginRequest(BaseModel):
	username: str
	password: str


class TokenResponse(BaseModel):
	access_token: str
	token_type: str = "bearer"
	role: str = "user"


class PredictionItem(BaseModel):
	label: str
	probability: float
	precautions: list[str]


class PredictionResponse(BaseModel):
	predictions: List[PredictionItem]


class UserResponse(BaseModel):
	email: str
	full_name: str
	role: str


