import os
import uuid
import shutil
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jwt
from passlib.context import CryptContext
import numpy as np
import soundfile as sf
import torchaudio
import torch

# Import Chatterbox TTS (real implementation only)
try:
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
    print("✅ Real Chatterbox TTS loaded successfully")
except ImportError as e:
    CHATTERBOX_AVAILABLE = False
    print(f"❌ Chatterbox TTS not available: {e}")
    print("Install from GitHub: pip install git+https://github.com/resemble-ai/chatterbox.git")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-this-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
PORT = int(os.getenv("PORT", "8000"))

# Initialize FastAPI
app = FastAPI(
    title="Chatterbox TTS API with JWT",
    description="Production-ready Text-to-Speech API with JWT authentication",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Storage directories
VOICES_DIR = Path("voices")
AUDIO_DIR = Path("audio")
VOICES_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# In-memory storage (use a database in production)
users_db: Dict[str, dict] = {}
voices_db: Dict[str, dict] = {}
audio_db: Dict[str, dict] = {}

# Initialize TTS model
class ChatterboxTTSWrapper:
    def __init__(self):
        self.model = None
        self.device = "cpu"  # Render typically doesn't have GPU
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Chatterbox TTS model"""
        try:
            if not CHATTERBOX_AVAILABLE:
                logger.error("Chatterbox TTS not available - cannot initialize")
                return
                
            # Try to use GPU if available, fallback to CPU
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA GPU for TTS")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Apple Silicon) for TTS")
            else:
                self.device = "cpu"
                logger.info("Using CPU for TTS (slower but works on Render)")
            
            logger.info(f"Initializing Chatterbox TTS on device: {self.device}")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("✅ Real Chatterbox TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chatterbox TTS: {e}")
            self.model = None
    
    def generate_speech(
        self, 
        text: str, 
        voice_id: str = "default",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        audio_prompt_path: Optional[str] = None
    ) -> tuple[np.ndarray, int]:
        """Generate speech using Chatterbox TTS"""
        if not self.model:
            raise Exception("Chatterbox TTS model not initialized")
            
        try:
            # Clean up text first
            text = text.strip()
            if not text:
                raise ValueError("Text cannot be empty")
            
            # Limit text length to avoid issues
            if len(text) > 1000:
                text = text[:1000]
                logger.warning("Text truncated to 1000 characters")
            
            logger.info(f"Generating speech for text: {text[:50]}...")
            
            # Generate speech with Chatterbox TTS
            if audio_prompt_path and Path(audio_prompt_path).exists():
                logger.info(f"Using custom voice from: {audio_prompt_path}")
                wav = self.model.generate(
                    text, 
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
            else:
                logger.info("Using default Chatterbox voice")
                wav = self.model.generate(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
            
            # Convert tensor to numpy if needed
            if torch.is_tensor(wav):
                wav = wav.cpu().numpy()
            
            # Ensure proper shape (1D audio)
            if wav.ndim > 1:
                wav = wav.squeeze()
            
            # Get sample rate from model
            sample_rate = getattr(self.model, 'sr', 24000)
            
            logger.info(f"✅ Speech generated successfully - {len(wav)} samples at {sample_rate}Hz")
            
            return wav, sample_rate
            
        except Exception as e:
            logger.error(f"Chatterbox TTS generation failed: {e}")
            raise Exception(f"Speech generation failed: {str(e)}")

# Initialize TTS
tts_engine = ChatterboxTTSWrapper()

# Pydantic models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = Field(None, max_length=100)

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=1000)
    voice_id: str = Field(default="default")
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.8, ge=0.05, le=5.0)

class VoiceResponse(BaseModel):
    voice_id: str
    name: str
    description: str
    type: str
    created_at: str
    audio_duration: Optional[float] = None

# Utility functions
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds"""
    try:
        data, samplerate = sf.read(file_path)
        return len(data) / samplerate
    except:
        return 0.0

# Routes
@app.get("/")
async def root():
    return {
        "message": "Chatterbox TTS API with JWT Authentication",
        "version": "1.0.0",
        "status": "running",
        "tts_available": tts_engine.model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "tts_model": "available" if tts_engine.model else "unavailable"
    }

@app.post("/auth/register", response_model=dict)
async def register(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_password = pwd_context.hash(user.password)
    users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat()
    }
    
    return {"message": "User registered successfully"}

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    if user.username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    stored_user = users_db[user.username]
    if not pwd_context.verify(user.password, stored_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.get("/profile")
async def get_profile(current_user: str = Depends(verify_token)):
    if current_user not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_data = users_db[current_user].copy()
    user_data.pop("hashed_password", None)
    return user_data

@app.get("/voices", response_model=dict)
async def list_voices(current_user: str = Depends(verify_token)):
    user_voices = [
        voice for voice in voices_db.values() 
        if voice.get("owner") == current_user
    ]
    
    # Add default voice
    default_voice = {
        "voice_id": "default",
        "name": "Default Voice",
        "description": "Built-in Chatterbox voice",
        "type": "builtin",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    all_voices = [default_voice] + user_voices
    
    return {
        "voices": all_voices,
        "total": len(all_voices),
        "builtin": 1,
        "custom": len(user_voices)
    }

@app.post("/voices", response_model=dict)
async def create_voice(
    voice_name: str = Form(...),
    voice_description: str = Form(...),
    audio_file: UploadFile = File(...),
    current_user: str = Depends(verify_token)
):
    # Validate file type
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an audio file"
        )
    
    # Generate unique voice ID
    voice_id = f"voice_{int(datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
    
    # Save audio file
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(exist_ok=True)
    
    audio_path = voice_dir / f"reference.{audio_file.filename.split('.')[-1]}"
    
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    # Get audio duration
    duration = get_audio_duration(str(audio_path))
    
    # Store voice metadata
    voice_data = {
        "voice_id": voice_id,
        "name": voice_name,
        "description": voice_description,
        "type": "custom",
        "owner": current_user,
        "audio_path": str(audio_path),
        "created_at": datetime.utcnow().isoformat(),
        "audio_duration": duration
    }
    
    voices_db[voice_id] = voice_data
    
    return {
        "success": True,
        "voice_id": voice_id,
        "message": f"Voice '{voice_name}' created successfully",
        "voice_info": {
            "voice_id": voice_id,
            "name": voice_name,
            "description": voice_description,
            "type": "custom",
            "created_at": voice_data["created_at"],
            "audio_duration": duration
        }
    }

@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str, current_user: str = Depends(verify_token)):
    if voice_id == "default":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete default voice"
        )
    
    if voice_id not in voices_db:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    voice = voices_db[voice_id]
    if voice.get("owner") != current_user:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to delete this voice"
        )
    
    # Delete files
    voice_dir = VOICES_DIR / voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
    
    # Remove from database
    del voices_db[voice_id]
    
    return {"message": f"Voice {voice_id} deleted successfully"}

@app.post("/synthesize", response_model=dict)
async def synthesize_speech(
    request: SynthesizeRequest,
    current_user: str = Depends(verify_token)
):
    if not tts_engine.model:
        raise HTTPException(
            status_code=500,
            detail="TTS model not available"
        )
    
    try:
        # Get voice path if custom voice
        audio_prompt_path = None
        if request.voice_id != "default":
            if request.voice_id not in voices_db:
                raise HTTPException(status_code=404, detail="Voice not found")
            
            voice = voices_db[request.voice_id]
            if voice.get("owner") != current_user:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to use this voice"
                )
            
            audio_prompt_path = voice["audio_path"]
        
        # Generate speech
        wav, sample_rate = tts_engine.generate_speech(
            text=request.text,
            voice_id=request.voice_id,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            audio_prompt_path=audio_prompt_path
        )
        
        # Save audio file
        audio_id = f"audio_{uuid.uuid4().hex}"
        audio_path = AUDIO_DIR / f"{audio_id}.wav"
        
        # Save using soundfile for better compatibility
        sf.write(str(audio_path), wav, sample_rate)
        
        # Store audio metadata
        duration = len(wav) / sample_rate
        audio_data = {
            "audio_id": audio_id,
            "owner": current_user,
            "voice_id": request.voice_id,
            "text": request.text,
            "file_path": str(audio_path),
            "sample_rate": sample_rate,
            "duration": duration,
            "created_at": datetime.utcnow().isoformat()
        }
        
        audio_db[audio_id] = audio_data
        
        return {
            "success": True,
            "audio_id": audio_id,
            "message": f"Speech synthesized successfully using voice '{request.voice_id}'",
            "sample_rate": sample_rate,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Speech synthesis failed: {str(e)}"
        )

@app.get("/audio/{audio_id}")
async def download_audio(audio_id: str, current_user: str = Depends(verify_token)):
    if audio_id not in audio_db:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio = audio_db[audio_id]
    if audio.get("owner") != current_user:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this audio"
        )
    
    file_path = audio["file_path"]
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"{audio_id}.wav"
    )

@app.get("/audio/{audio_id}/info")
async def get_audio_info(audio_id: str, current_user: str = Depends(verify_token)):
    if audio_id not in audio_db:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio = audio_db[audio_id]
    if audio.get("owner") != current_user:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this audio"
        )
    
    # Remove file path from response
    audio_info = audio.copy()
    audio_info.pop("file_path", None)
    return audio_info

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False
    )
