# Simple working main.py for Render
import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chatterbox TTS API with JWT Authentication",
    description="Production-ready Text-to-Speech API with JWT authentication",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-jwt-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# In-memory storage for demo
users_db = {}
voices_db = {
    "default": {
        "voice_id": "default",
        "name": "Default Voice",
        "description": "Default system voice",
        "type": "builtin",
        "created_at": datetime.utcnow().isoformat(),
        "user_id": "system"
    }
}

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "default"

# Helper functions
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except Exception:
        raise credentials_exception
    
    user = users_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    return user

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Chatterbox TTS API with JWT Authentication",
        "version": "2.0.0",
        "status": "running",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@app.post("/auth/register", response_model=dict)
async def register(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check email
    for existing_user in users_db.values():
        if existing_user.get("email") == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed_password = get_password_hash(user.password)
    users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "is_active": True
    }
    
    return {"message": "User registered successfully", "username": user.username}

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    if user.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    user_data = users_db[user.username]
    if not verify_password(user.password, user_data["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get("/voices")
async def list_voices(current_user: dict = Depends(get_current_user)):
    return {
        "voices": list(voices_db.values()),
        "total": len(voices_db),
        "builtin": 1,
        "custom": 0
    }

@app.post("/synthesize")
async def synthesize_speech(
    request: SynthesizeRequest,
    current_user: dict = Depends(get_current_user)
):
    if request.voice_id not in voices_db:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    # Generate simple demo response
    audio_id = f"audio_{int(datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
    
    return {
        "success": True,
        "audio_id": audio_id,
        "message": f"Speech synthesized successfully (demo mode)",
        "text": request.text,
        "voice_id": request.voice_id,
        "note": "This is demo mode - no actual audio generated"
    }

@app.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "full_name": current_user.get("full_name"),
        "created_at": current_user["created_at"],
        "is_active": current_user.get("is_active", True)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
