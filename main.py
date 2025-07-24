# main.py - FastAPI application with JWT authentication
import os
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import base64
import io
import wave

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext
import torch
import torchaudio
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chatterbox TTS API with JWT Authentication",
    description="Production-ready Text-to-Speech API with voice cloning and JWT authentication",
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
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Storage directories
VOICES_DIR = "voices"
AUDIO_DIR = "audio"
USERS_FILE = "users.json"

# Create directories
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Device configuration (CPU only for Render)
device = "cpu"
logger.info(f"Using device: {device}")

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
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None

class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    description: str
    type: str
    created_at: str
    user_id: str
    audio_duration: Optional[float] = None

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speed: float = 1.0
    pitch: float = 1.0

class AudioInfo(BaseModel):
    audio_id: str
    user_id: str
    voice_id: str
    text: str
    created_at: str
    duration: float
    sample_rate: int

# Mock TTS class for CPU (since ChatterboxTTS requires GPU)
class MockTTS:
    def __init__(self, device="cpu"):
        self.device = device
        logger.info(f"Initialized Mock TTS on device: {device}")
    
    def generate_speech(self, text: str, voice_id: str = "default", **kwargs) -> tuple:
        """Generate mock speech for demo purposes"""
        # Generate a simple sine wave as placeholder audio
        duration = len(text) * 0.1  # 0.1 seconds per character
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Different frequencies for different voices
        voice_frequencies = {
            "default": 440,
            "female_default": 523,
            "male_default": 349
        }
        frequency = voice_frequencies.get(voice_id, 440)
        
        # Generate sine wave
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Convert to int16
        audio = (audio * 32767).astype(np.int16)
        
        return audio, sample_rate

# Initialize TTS model
try:
    # Try to import and use actual ChatterboxTTS if available
    from chatterbox.src.chatterbox.tts import ChatterboxTTS
    tts_model = ChatterboxTTS(device=device)
    logger.info("Loaded ChatterboxTTS model")
except Exception as e:
    logger.warning(f"Could not load ChatterboxTTS: {e}. Using mock TTS.")
    tts_model = MockTTS(device=device)

# User management functions
def load_users() -> Dict[str, Any]:
    """Load users from JSON file"""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users: Dict[str, Any]):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
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
    except jwt.PyJWTError:
        raise credentials_exception
    
    users = load_users()
    user = users.get(token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Voice management functions
def load_voices() -> Dict[str, Any]:
    """Load voices metadata"""
    voices_file = os.path.join(VOICES_DIR, "voices.json")
    try:
        with open(voices_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Initialize with default voices
        default_voices = {
            "default": {
                "voice_id": "default",
                "name": "Default Voice",
                "description": "Default system voice",
                "type": "builtin",
                "created_at": datetime.utcnow().isoformat(),
                "user_id": "system"
            },
            "female_default": {
                "voice_id": "female_default",
                "name": "Female Default",
                "description": "Default female voice",
                "type": "builtin",
                "created_at": datetime.utcnow().isoformat(),
                "user_id": "system"
            },
            "male_default": {
                "voice_id": "male_default",
                "name": "Male Default",
                "description": "Default male voice",
                "type": "builtin",
                "created_at": datetime.utcnow().isoformat(),
                "user_id": "system"
            }
        }
        save_voices(default_voices)
        return default_voices

def save_voices(voices: Dict[str, Any]):
    """Save voices metadata"""
    voices_file = os.path.join(VOICES_DIR, "voices.json")
    with open(voices_file, 'w') as f:
        json.dump(voices, f, indent=2)

def save_audio_to_file(audio_data: np.ndarray, sample_rate: int, audio_id: str) -> str:
    """Save audio data to WAV file"""
    audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
    
    # Create WAV file
    with wave.open(audio_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return audio_path

# API Routes

@app.get("/", tags=["Root"])
async def root():
    """API status and documentation"""
    return {
        "message": "Chatterbox TTS API with JWT Authentication",
        "version": "2.0.0",
        "status": "running",
        "device": device,
        "documentation": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "device": device,
        "version": "2.0.0"
    }

# Authentication endpoints
@app.post("/auth/register", response_model=dict, tags=["Authentication"])
async def register(user: UserCreate):
    """Register a new user"""
    users = load_users()
    
    if user.username in users:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Check if email already exists
    for existing_user in users.values():
        if existing_user.get("email") == user.email:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    users[user.username] = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "is_active": True
    }
    
    save_users(users)
    
    return {
        "message": "User registered successfully",
        "username": user.username
    }

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(user: UserLogin):
    """Login and get JWT tokens"""
    users = load_users()
    
    if user.username not in users:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )
    
    user_data = users[user.username]
    if not verify_password(user.password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )
    
    if not user_data.get("is_active", True):
        raise HTTPException(
            status_code=401,
            detail="User account is deactivated"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.post("/auth/refresh", response_model=Token, tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """Refresh JWT tokens"""
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    new_refresh_token = create_refresh_token(data={"sub": username})
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }

# Voice endpoints
@app.get("/voices", response_model=dict, tags=["Voices"])
async def list_voices(current_user: dict = Depends(get_current_user)):
    """List all available voices"""
    voices = load_voices()
    
    # Filter voices (user can see builtin + their own voices)
    user_voices = {}
    for voice_id, voice_data in voices.items():
        if voice_data.get("type") == "builtin" or voice_data.get("user_id") == current_user["username"]:
            user_voices[voice_id] = voice_data
    
    return {
        "voices": list(user_voices.values()),
        "total": len(user_voices),
        "builtin": len([v for v in user_voices.values() if v.get("type") == "builtin"]),
        "custom": len([v for v in user_voices.values() if v.get("type") == "custom"])
    }

@app.post("/voices", response_model=dict, tags=["Voices"])
async def create_voice(
    voice_name: str = Form(...),
    voice_description: str = Form(...),
    audio_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Create a new voice from audio file"""
    
    # Validate audio file
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Generate voice ID
    voice_id = f"voice_{int(datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
    
    # Save audio file
    audio_content = await audio_file.read()
    audio_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    
    with open(audio_path, "wb") as f:
        f.write(audio_content)
    
    # Calculate audio duration (placeholder)
    try:
        # Try to get actual duration
        import wave
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)
    except:
        duration = 10.0  # Default duration
    
    # Save voice metadata
    voices = load_voices()
    voice_info = {
        "voice_id": voice_id,
        "name": voice_name,
        "description": voice_description,
        "type": "custom",
        "created_at": datetime.utcnow().isoformat(),
        "user_id": current_user["username"],
        "audio_duration": duration,
        "audio_path": audio_path
    }
    
    voices[voice_id] = voice_info
    save_voices(voices)
    
    return {
        "success": True,
        "voice_id": voice_id,
        "message": f"Voice '{voice_name}' created successfully",
        "voice_info": voice_info
    }

@app.delete("/voices/{voice_id}", response_model=dict, tags=["Voices"])
async def delete_voice(voice_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a voice"""
    voices = load_voices()
    
    if voice_id not in voices:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    voice_data = voices[voice_id]
    
    # Check permissions
    if voice_data.get("type") == "builtin":
        raise HTTPException(status_code=403, detail="Cannot delete builtin voices")
    
    if voice_data.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only delete your own voices")
    
    # Delete audio file
    audio_path = voice_data.get("audio_path")
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    
    # Remove from metadata
    del voices[voice_id]
    save_voices(voices)
    
    return {
        "success": True,
        "message": f"Voice '{voice_data['name']}' deleted successfully"
    }

# Speech synthesis endpoints
@app.post("/synthesize", response_model=dict, tags=["Speech Synthesis"])
async def synthesize_speech(
    request: SynthesizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate speech from text using specified voice"""
    
    # Validate voice
    voices = load_voices()
    if request.voice_id not in voices:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    voice_data = voices[request.voice_id]
    
    # Check permissions for custom voices
    if voice_data.get("type") == "custom" and voice_data.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Cannot use other users' custom voices")
    
    try:
        # Generate speech
        audio_data, sample_rate = tts_model.generate_speech(
            text=request.text,
            voice_id=request.voice_id,
            speed=request.speed,
            pitch=request.pitch
        )
        
        # Generate audio ID
        audio_id = f"audio_{int(datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        # Save audio file
        audio_path = save_audio_to_file(audio_data, sample_rate, audio_id)
        
        # Calculate duration
        duration = len(audio_data) / sample_rate
        
        # Save audio metadata
        audio_info = {
            "audio_id": audio_id,
            "user_id": current_user["username"],
            "voice_id": request.voice_id,
            "text": request.text,
            "created_at": datetime.utcnow().isoformat(),
            "duration": duration,
            "sample_rate": sample_rate,
            "audio_path": audio_path
        }
        
        # Save to audio metadata file
        audio_metadata_file = os.path.join(AUDIO_DIR, "audio_metadata.json")
        try:
            with open(audio_metadata_file, 'r') as f:
                audio_metadata = json.load(f)
        except FileNotFoundError:
            audio_metadata = {}
        
        audio_metadata[audio_id] = audio_info
        
        with open(audio_metadata_file, 'w') as f:
            json.dump(audio_metadata, f, indent=2)
        
        return {
            "success": True,
            "audio_id": audio_id,
            "message": f"Speech synthesized successfully using voice '{voice_data['name']}'",
            "sample_rate": sample_rate,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Speech synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.get("/audio/{audio_id}", tags=["Audio"])
async def get_audio(audio_id: str, current_user: dict = Depends(get_current_user)):
    """Download generated audio file"""
    
    # Load audio metadata
    audio_metadata_file = os.path.join(AUDIO_DIR, "audio_metadata.json")
    try:
        with open(audio_metadata_file, 'r') as f:
            audio_metadata = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    if audio_id not in audio_metadata:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_info = audio_metadata[audio_id]
    
    # Check permissions
    if audio_info.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only access your own audio files")
    
    audio_path = audio_info.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"{audio_id}.wav"
    )

@app.get("/audio/{audio_id}/info", response_model=AudioInfo, tags=["Audio"])
async def get_audio_info(audio_id: str, current_user: dict = Depends(get_current_user)):
    """Get audio metadata"""
    
    # Load audio metadata
    audio_metadata_file = os.path.join(AUDIO_DIR, "audio_metadata.json")
    try:
        with open(audio_metadata_file, 'r') as f:
            audio_metadata = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    if audio_id not in audio_metadata:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_info = audio_metadata[audio_id]
    
    # Check permissions
    if audio_info.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only access your own audio files")
    
    return AudioInfo(**audio_info)

# User profile endpoints
@app.get("/profile", tags=["User"])
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
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
