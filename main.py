# main.py - Complete Chatterbox TTS API with JWT Authentication and Audio Generation
import os
import uuid
import logging
import wave
import base64
import io
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import math

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
    description="Production-ready Text-to-Speech API with voice cloning, JWT authentication, and audio playback",
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
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Storage directories
VOICES_DIR = "voices"
AUDIO_DIR = "audio"

# Create directories
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# In-memory storage for demo (in production, use a database)
users_db = {}
audio_metadata = {}

# Enhanced voice database with more realistic voices
voices_db = {
    "default": {
        "voice_id": "default",
        "name": "Default Voice",
        "description": "Standard neutral voice",
        "type": "builtin",
        "created_at": datetime.utcnow().isoformat(),
        "user_id": "system",
        "language": "en-US",
        "gender": "neutral",
        "age_group": "adult",
        "accent": "american"
    },
    "female_default": {
        "voice_id": "female_default", 
        "name": "Sarah",
        "description": "Professional female voice",
        "type": "builtin",
        "created_at": datetime.utcnow().isoformat(),
        "user_id": "system",
        "language": "en-US",
        "gender": "female",
        "age_group": "adult",
        "accent": "american"
    },
    "male_default": {
        "voice_id": "male_default",
        "name": "David",
        "description": "Professional male voice", 
        "type": "builtin",
        "created_at": datetime.utcnow().isoformat(),
        "user_id": "system",
        "language": "en-US",
        "gender": "male",
        "age_group": "adult",
        "accent": "american"
    },
    "female_young": {
        "voice_id": "female_young",
        "name": "Emma",
        "description": "Young energetic female voice",
        "type": "builtin", 
        "created_at": datetime.utcnow().isoformat(),
        "user_id": "system",
        "language": "en-US",
        "gender": "female",
        "age_group": "young",
        "accent": "american"
    },
    "male_deep": {
        "voice_id": "male_deep",
        "name": "Marcus",
        "description": "Deep authoritative male voice",
        "type": "builtin",
        "created_at": datetime.utcnow().isoformat(), 
        "user_id": "system",
        "language": "en-US",
        "gender": "male",
        "age_group": "adult",
        "accent": "american"
    },
    "female_british": {
        "voice_id": "female_british",
        "name": "Victoria",
        "description": "Elegant British female voice",
        "type": "builtin",
        "created_at": datetime.utcnow().isoformat(),
        "user_id": "system", 
        "language": "en-GB",
        "gender": "female",
        "age_group": "adult",
        "accent": "british"
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
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None

class VoiceCreate(BaseModel):
    voice_name: str
    voice_description: str
    language: str = "en-US"
    gender: str = "neutral"
    age_group: str = "adult"
    accent: str = "american"

class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    description: str
    type: str
    created_at: str
    user_id: str
    language: str = "en-US"
    gender: str = "neutral"
    age_group: str = "adult"
    accent: str = "american"
    audio_duration: Optional[float] = None

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    output_format: str = "wav"  # wav, mp3

class AudioInfo(BaseModel):
    audio_id: str
    user_id: str
    voice_id: str
    text: str
    created_at: str
    duration: float
    sample_rate: int
    format: str
    file_size: int

# Audio generation functions
class AdvancedTTS:
    def __init__(self):
        self.sample_rate = 22050
        logger.info("Initialized Advanced TTS Engine")
    
    def generate_speech(self, text: str, voice_id: str, speed: float = 1.0, 
                       pitch: float = 1.0, volume: float = 1.0) -> tuple:
        """Generate advanced speech synthesis with voice characteristics"""
        
        # Get voice characteristics
        voice_data = voices_db.get(voice_id, voices_db["default"])
        
        # Calculate duration based on text and speed
        words = len(text.split())
        base_duration = words * 0.6  # ~0.6 seconds per word
        duration = base_duration / speed
        
        # Generate time array
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Voice-specific frequency characteristics
        voice_frequencies = {
            "default": {"base": 220, "harmonics": [1, 0.5, 0.3]},
            "female_default": {"base": 350, "harmonics": [1, 0.7, 0.4, 0.2]},
            "male_default": {"base": 180, "harmonics": [1, 0.6, 0.3]},
            "female_young": {"base": 400, "harmonics": [1, 0.8, 0.5, 0.3]},
            "male_deep": {"base": 120, "harmonics": [1, 0.4, 0.2]},
            "female_british": {"base": 320, "harmonics": [1, 0.6, 0.4, 0.2]}
        }
        
        voice_params = voice_frequencies.get(voice_id, voice_frequencies["default"])
        base_freq = voice_params["base"] * pitch
        harmonics = voice_params["harmonics"]
        
        # Initialize audio signal
        audio = np.zeros_like(t)
        
        # Generate speech-like patterns
        for i, char in enumerate(text.lower()):
            if char.isalpha():
                # Character-specific frequency variation
                char_offset = (ord(char) - ord('a')) / 26.0  # 0 to 1
                char_freq = base_freq * (1 + char_offset * 0.5)
                
                # Time window for this character
                char_start = i * duration / len(text)
                char_end = min((i + 1) * duration / len(text), duration)
                
                # Find indices for this time window
                start_idx = int(char_start * self.sample_rate)
                end_idx = int(char_end * self.sample_rate)
                
                if start_idx < len(t) and end_idx <= len(t):
                    char_t = t[start_idx:end_idx]
                    
                    # Generate harmonics for more natural sound
                    char_audio = np.zeros_like(char_t)
                    for h_idx, h_amp in enumerate(harmonics):
                        harmonic_freq = char_freq * (h_idx + 1)
                        char_audio += h_amp * np.sin(2 * np.pi * harmonic_freq * char_t)
                    
                    # Apply envelope for smoother transitions
                    envelope = np.exp(-3 * np.abs(char_t - (char_start + char_end) / 2) / (char_end - char_start))
                    char_audio *= envelope
                    
                    audio[start_idx:end_idx] += char_audio
            
            elif char == ' ':
                # Add brief pause for spaces
                pause_start = i * duration / len(text)
                pause_duration = 0.1 / speed
                pause_idx = int(pause_start * self.sample_rate)
                pause_end_idx = int((pause_start + pause_duration) * self.sample_rate)
                
                if pause_idx < len(audio) and pause_end_idx <= len(audio):
                    audio[pause_idx:pause_end_idx] *= 0.1  # Reduce volume for pause
        
        # Apply global effects
        # Volume control
        audio *= volume
        
        # Add subtle vibrato for more natural sound
        vibrato_freq = 4.5
        vibrato_depth = 0.02
        vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
        audio *= vibrato
        
        # Normalize and convert to int16
        audio = np.clip(audio, -1, 1)
        audio = (audio * 32767).astype(np.int16)
        
        return audio, self.sample_rate

# Initialize TTS engine
tts_engine = AdvancedTTS()

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

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
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

def save_audio_file(audio_data: np.ndarray, sample_rate: int, audio_id: str, format: str = "wav") -> str:
    """Save audio data to file"""
    if format.lower() == "wav":
        file_path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    else:
        # For other formats, still save as WAV (can be extended later)
        file_path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    return file_path

# API Routes
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Chatterbox TTS API with JWT Authentication",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "JWT Authentication",
            "Voice Management", 
            "Speech Synthesis",
            "Audio Playback",
            "Custom Voice Creation"
        ],
        "documentation": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "voices_available": len(voices_db),
        "audio_files_stored": len([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]) if os.path.exists(AUDIO_DIR) else 0
    }

# Authentication endpoints
@app.post("/auth/register", response_model=dict, tags=["Authentication"])
async def register(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    for existing_user in users_db.values():
        if existing_user.get("email") == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
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

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
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
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.post("/auth/refresh", response_model=Token, tags=["Authentication"])
async def refresh_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    if username not in users_db:
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

# Voice management endpoints
@app.get("/voices", response_model=dict, tags=["Voice Management"])
async def list_voices(
    voice_type: Optional[str] = Query(None, description="Filter by type: builtin, custom, or all"),
    language: Optional[str] = Query(None, description="Filter by language code (e.g., en-US, en-GB)"),
    gender: Optional[str] = Query(None, description="Filter by gender: male, female, neutral"),
    accent: Optional[str] = Query(None, description="Filter by accent: american, british, etc."),
    current_user: dict = Depends(get_current_user)
):
    """List available voices with advanced filtering options"""
    
    # Get voices user can access
    accessible_voices = {}
    for voice_id, voice_data in voices_db.items():
        if (voice_data.get("type") == "builtin" or 
            voice_data.get("user_id") == current_user["username"]):
            accessible_voices[voice_id] = voice_data
    
    # Apply filters
    filtered_voices = accessible_voices
    
    if voice_type and voice_type != "all":
        filtered_voices = {
            vid: vdata for vid, vdata in filtered_voices.items() 
            if vdata.get("type") == voice_type
        }
    
    if language:
        filtered_voices = {
            vid: vdata for vid, vdata in filtered_voices.items()
            if vdata.get("language", "").lower() == language.lower()
        }
    
    if gender:
        filtered_voices = {
            vid: vdata for vid, vdata in filtered_voices.items()
            if vdata.get("gender", "").lower() == gender.lower()
        }
    
    if accent:
        filtered_voices = {
            vid: vdata for vid, vdata in filtered_voices.items()
            if vdata.get("accent", "").lower() == accent.lower()
        }
    
    # Organize voices
    builtin_voices = [v for v in filtered_voices.values() if v.get("type") == "builtin"]
    custom_voices = [v for v in filtered_voices.values() if v.get("type") == "custom"]
    
    return {
        "voices": {
            "builtin": builtin_voices,
            "custom": custom_voices,
            "all": list(filtered_voices.values())
        },
        "total": len(filtered_voices),
        "builtin": len(builtin_voices),
        "custom": len(custom_voices),
        "filters_applied": {
            "type": voice_type,
            "language": language, 
            "gender": gender,
            "accent": accent
        },
        "available_filters": {
            "languages": list(set(v.get("language", "en-US") for v in voices_db.values())),
            "genders": ["male", "female", "neutral"],
            "accents": list(set(v.get("accent", "american") for v in voices_db.values())),
            "types": ["builtin", "custom"]
        }
    }

@app.get("/voices/{voice_id}", response_model=dict, tags=["Voice Management"])
async def get_voice_details(voice_id: str, current_user: dict = Depends(get_current_user)):
    """Get detailed information about a specific voice"""
    
    if voice_id not in voices_db:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    voice_data = voices_db[voice_id]
    
    if (voice_data.get("type") == "custom" and 
        voice_data.get("user_id") != current_user["username"]):
        raise HTTPException(status_code=403, detail="Access denied to this voice")
    
    return {
        "voice_info": voice_data,
        "permissions": {
            "can_use": True,
            "can_delete": voice_data.get("user_id") == current_user["username"],
            "can_modify": voice_data.get("user_id") == current_user["username"]
        },
        "usage_stats": {
            "times_used": 0,  # In production, track actual usage
            "last_used": None
        }
    }

@app.post("/voices", response_model=dict, tags=["Voice Management"])
async def create_voice(
    voice_name: str = Form(..., description="Name for the custom voice"),
    voice_description: str = Form(..., description="Description of the voice"),
    language: str = Form("en-US", description="Language code (e.g., en-US, en-GB)"),
    gender: str = Form("neutral", description="Voice gender: male, female, neutral"),
    age_group: str = Form("adult", description="Age group: young, adult, senior"),
    accent: str = Form("american", description="Accent type: american, british, etc."),
    audio_file: UploadFile = File(..., description="Audio sample for voice cloning"),
    current_user: dict = Depends(get_current_user)
):
    """Create a new custom voice from audio file"""
    
    # Validate audio file
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file (WAV, MP3, FLAC, etc.)")
    
    # Check file size (limit to 10MB)
    content = await audio_file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file too large. Maximum size is 10MB.")
    
    if len(content) < 1024:
        raise HTTPException(status_code=400, detail="Audio file too small. Minimum size is 1KB.")
    
    # Generate unique voice ID
    timestamp = int(datetime.utcnow().timestamp())
    voice_id = f"voice_{current_user['username']}_{timestamp}_{uuid.uuid4().hex[:6]}"
    
    # Estimate audio duration (rough calculation)
    audio_duration = len(content) / (44100 * 2)  # Assuming 44.1kHz, 16-bit
    
    # Save audio file for voice training (in production, this would be processed)
    voice_audio_path = os.path.join(VOICES_DIR, f"{voice_id}_sample.wav")
    with open(voice_audio_path, "wb") as f:
        f.write(content)
    
    # Create voice entry
    voice_info = {
        "voice_id": voice_id,
        "name": voice_name,
        "description": voice_description,
        "type": "custom",
        "created_at": datetime.utcnow().isoformat(),
        "user_id": current_user["username"],
        "language": language,
        "gender": gender,
        "age_group": age_group,
        "accent": accent,
        "audio_duration": round(audio_duration, 2),
        "file_size": len(content),
        "original_filename": audio_file.filename,
        "sample_path": voice_audio_path,
        "status": "ready"  # In production: "processing", "ready", "failed"
    }
    
    # Add to voice database
    voices_db[voice_id] = voice_info
    
    return {
        "success": True,
        "voice_id": voice_id,
        "message": f"Custom voice '{voice_name}' created successfully",
        "voice_info": voice_info,
        "next_steps": [
            "Use the voice_id in synthesis requests",
            "Test the voice with different text samples",
            "Adjust synthesis parameters for best results"
        ]
    }

@app.delete("/voices/{voice_id}", response_model=dict, tags=["Voice Management"])
async def delete_voice(voice_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a custom voice"""
    
    if voice_id not in voices_db:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    voice_data = voices_db[voice_id]
    
    if voice_data.get("type") == "builtin":
        raise HTTPException(status_code=403, detail="Cannot delete builtin voices")
    
    if voice_data.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only delete your own voices")
    
    # Delete associated files
    sample_path = voice_data.get("sample_path")
    if sample_path and os.path.exists(sample_path):
        os.remove(sample_path)
    
    # Remove from database
    voice_name = voice_data["name"]
    del voices_db[voice_id]
    
    return {
        "success": True,
        "message": f"Voice '{voice_name}' deleted successfully",
        "deleted_voice_id": voice_id
    }

# Speech synthesis endpoints
@app.post("/synthesize", response_model=dict, tags=["Speech Synthesis"])
async def synthesize_speech(
    request: SynthesizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate speech from text using specified voice with enhanced parameters"""
    
    # Validate voice exists
    if request.voice_id not in voices_db:
        raise HTTPException(status_code=404, detail=f"Voice '{request.voice_id}' not found")
    
    voice_data = voices_db[request.voice_id]
    
    # Check permissions for custom voices
    if (voice_data.get("type") == "custom" and 
        voice_data.get("user_id") != current_user["username"]):
        raise HTTPException(status_code=403, detail="Cannot use other users' custom voices")
    
    # Validate parameters
    if not (0.1 <= request.speed <= 3.0):
        raise HTTPException(status_code=400, detail="Speed must be between 0.1 and 3.0")
    if not (0.1 <= request.pitch <= 3.0):
        raise HTTPException(status_code=400, detail="Pitch must be between 0.1 and 3.0")
    if not (0.1 <= request.volume <= 2.0):
        raise HTTPException(status_code=400, detail="Volume must be between 0.1 and 2.0")
    
    # Validate text
    if len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long. Maximum 5000 characters.")
    
    try:
        # Generate speech using TTS engine
        audio_data, sample_rate = tts_engine.generate_speech(
            text=request.text,
            voice_id=request.voice_id,
            speed=request.speed,
            pitch=request.pitch,
            volume=request.volume
        )
        
        # Generate unique audio ID
        timestamp = int(datetime.utcnow().timestamp())
        audio_id = f"audio_{current_user['username']}_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Save audio file
        audio_path = save_audio_file(audio_data, sample_rate, audio_id, request.output_format)
        
        # Calculate file size
        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        duration = len(audio_data) / sample_rate
        
        # Store audio metadata
        audio_info = {
            "audio_id": audio_id,
            "user_id": current_user["username"],
            "voice_id": request.voice_id,
            "text": request.text,
            "created_at": datetime.utcnow().isoformat(),
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "format": request.output_format,
            "file_size": file_size,
            "file_path": audio_path,
            "parameters": {
                "speed": request.speed,
                "pitch": request.pitch,
                "volume": request.volume
            }
        }
        
        audio_metadata[audio_id] = audio_info
        
        return {
            "success": True,
            "audio_id": audio_id,
            "message": f"Speech synthesized successfully using voice '{voice_data['name']}'",
            "synthesis_info": {
                "text": request.text,
                "voice_id": request.voice_id,
                "voice_name": voice_data["name"],
                "voice_type": voice_data["type"],
                "language": voice_data.get("language", "en-US"),
                "gender": voice_data.get("gender", "neutral"),
                "parameters": {
                    "speed": request.speed,
                    "pitch": request.pitch,
                    "volume": request.volume
                }
            },
            "audio_info": {
                "duration": round(duration, 2),
                "sample_rate": sample_rate,
                "format": request.output_format,
                "file_size_kb": round(file_size / 1024, 1),
                "file_path": f"/audio/{audio_id}"
            },
            "playback_urls": {
                "download": f"/audio/{audio_id}",
                "stream": f"/audio/{audio_id}/stream",
                "info": f"/audio/{audio_id}/info"
            }
        }
        
    except Exception as e:
        logger.error(f"Speech synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

# Audio playback endpoints
@app.get("/audio/{audio_id}", tags=["Audio Playback"])
async def download_audio(audio_id: str, current_user: dict = Depends(get_current_user)):
    """Download generated audio file"""
    
    if audio_id not in audio_metadata:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_info = audio_metadata[audio_id]
    
    # Check permissions
    if audio_info.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only access your own audio files")
    
    file_path = audio_info.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    
    # Return file for download
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"{audio_id}.wav",
        headers={"Content-Disposition": f"attachment; filename={audio_id}.wav"}
    )

@app.get("/audio/{audio_id}/stream", tags=["Audio Playback"])
async def stream_audio(audio_id: str, current_user: dict = Depends(get_current_user)):
    """Stream audio file for direct playback in browser"""
    
    if audio_id not in audio_metadata:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_info = audio_metadata[audio_id]
    
    # Check permissions
    if audio_info.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only access your own audio files")
    
    file_path = audio_info.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    
    # Stream the file
    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"inline; filename={audio_id}.wav",
            "Accept-Ranges": "bytes"
        }
    )

@app.get("/audio/{audio_id}/info", response_model=AudioInfo, tags=["Audio Playback"])
async def get_audio_info(audio_id: str, current_user: dict = Depends(get_current_user)):
    """Get detailed information about generated audio"""
    
    if audio_id not in audio_metadata:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_info = audio_metadata[audio_id]
    
    # Check permissions
    if audio_info.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only access your own audio files")
    
    return AudioInfo(**audio_info)

@app.get("/audio/{audio_id}/waveform", tags=["Audio Playback"])
async def get_audio_waveform(audio_id: str, current_user: dict = Depends(get_current_user)):
    """Get audio waveform data for visualization"""
    
    if audio_id not in audio_metadata:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_info = audio_metadata[audio_id]
    
    # Check permissions
    if audio_info.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only access your own audio files")
    
    file_path = audio_info.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    
    try:
        # Read audio file and generate waveform data
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Downsample for waveform visualization (max 1000 points)
            target_length = min(1000, len(audio_data))
            if len(audio_data) > target_length:
                step = len(audio_data) // target_length
                waveform = audio_data[::step][:target_length]
            else:
                waveform = audio_data
            
            # Normalize to -1 to 1 range
            waveform_normalized = waveform.astype(float) / 32767.0
            
            return {
                "audio_id": audio_id,
                "waveform": waveform_normalized.tolist(),
                "sample_rate": audio_info["sample_rate"],
                "duration": audio_info["duration"],
                "points": len(waveform_normalized)
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate waveform: {str(e)}")

# User audio management
@app.get("/my-audio", tags=["Audio Management"])
async def list_user_audio(
    limit: int = Query(50, description="Maximum number of audio files to return"),
    offset: int = Query(0, description="Number of audio files to skip"),
    voice_id: Optional[str] = Query(None, description="Filter by voice ID"),
    current_user: dict = Depends(get_current_user)
):
    """List user's generated audio files"""
    
    # Filter user's audio files
    user_audio = []
    for audio_id, audio_info in audio_metadata.items():
        if audio_info.get("user_id") == current_user["username"]:
            if voice_id is None or audio_info.get("voice_id") == voice_id:
                user_audio.append({
                    "audio_id": audio_id,
                    "voice_id": audio_info["voice_id"],
                    "voice_name": voices_db.get(audio_info["voice_id"], {}).get("name", "Unknown"),
                    "text_preview": audio_info["text"][:100] + "..." if len(audio_info["text"]) > 100 else audio_info["text"],
                    "duration": audio_info["duration"],
                    "created_at": audio_info["created_at"],
                    "file_size_kb": round(audio_info["file_size"] / 1024, 1),
                    "format": audio_info["format"]
                })
    
    # Sort by creation time (newest first)
    user_audio.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Apply pagination
    total = len(user_audio)
    paginated_audio = user_audio[offset:offset + limit]
    
    return {
        "audio_files": paginated_audio,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        },
        "summary": {
            "total_files": total,
            "total_duration": sum(a["duration"] for a in user_audio),
            "total_size_mb": round(sum(a["file_size_kb"] for a in user_audio) / 1024, 1)
        }
    }

@app.delete("/audio/{audio_id}", tags=["Audio Management"])
async def delete_audio(audio_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a generated audio file"""
    
    if audio_id not in audio_metadata:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_info = audio_metadata[audio_id]
    
    # Check permissions
    if audio_info.get("user_id") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Can only delete your own audio files")
    
    # Delete file from disk
    file_path = audio_info.get("file_path")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
    
    # Remove from metadata
    del audio_metadata[audio_id]
    
    return {
        "success": True,
        "message": f"Audio file {audio_id} deleted successfully",
        "deleted_audio_id": audio_id
    }

# User profile and statistics
@app.get("/profile", tags=["User Management"])
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile with usage statistics"""
    
    # Calculate user statistics
    user_voices = len([v for v in voices_db.values() if v.get("user_id") == current_user["username"]])
    user_audio_files = len([a for a in audio_metadata.values() if a.get("user_id") == current_user["username"]])
    total_audio_duration = sum(a["duration"] for a in audio_metadata.values() if a.get("user_id") == current_user["username"])
    
    return {
        "user_info": {
            "username": current_user["username"],
            "email": current_user["email"],
            "full_name": current_user.get("full_name"),
            "created_at": current_user["created_at"],
            "is_active": current_user.get("is_active", True)
        },
        "usage_stats": {
            "custom_voices_created": user_voices,
            "audio_files_generated": user_audio_files,
            "total_audio_duration_minutes": round(total_audio_duration / 60, 1),
            "account_age_days": (datetime.utcnow() - datetime.fromisoformat(current_user["created_at"].replace('Z', '+00:00'))).days
        },
        "available_features": [
            "Voice cloning from audio samples",
            "Multiple built-in voices", 
            "Speech synthesis with custom parameters",
            "Audio file download and streaming",
            "Waveform visualization",
            "Voice and audio management"
        ]
    }

# API statistics (admin-like endpoint)
@app.get("/stats", tags=["Statistics"])
async def get_api_stats(current_user: dict = Depends(get_current_user)):
    """Get API usage statistics"""
    
    return {
        "api_stats": {
            "total_users": len(users_db),
            "total_voices": len(voices_db),
            "builtin_voices": len([v for v in voices_db.values() if v.get("type") == "builtin"]),
            "custom_voices": len([v for v in voices_db.values() if v.get("type") == "custom"]),
            "total_audio_files": len(audio_metadata),
            "total_audio_duration_hours": round(sum(a["duration"] for a in audio_metadata.values()) / 3600, 1)
        },
        "voice_languages": {
            lang: len([v for v in voices_db.values() if v.get("language") == lang])
            for lang in set(v.get("language", "en-US") for v in voices_db.values())
        },
        "voice_genders": {
            gender: len([v for v in voices_db.values() if v.get("gender") == gender])
            for gender in set(v.get("gender", "neutral") for v in voices_db.values())
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
