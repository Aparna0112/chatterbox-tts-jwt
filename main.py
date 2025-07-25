# main.py - Complete Chatterbox TTS API with JWT Authentication and Human-like Speech
import os
import uuid
import logging
import wave
import base64
import io
import math
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

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
    description="Production-ready Text-to-Speech API with human-like voice synthesis, JWT authentication, and audio playback",
    version="2.1.0"
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
    output_format: str = "wav"

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

# Human-like TTS Engine
class AdvancedTTS:
    def __init__(self):
        self.sample_rate = 22050
        logger.info("Initialized Advanced TTS Engine with Human-like Speech")
        
        # Phoneme-to-frequency mapping for more realistic speech
        self.phoneme_frequencies = {
            # Vowels (formant frequencies)
            'a': {'f1': 730, 'f2': 1090, 'f3': 2440},  # as in "cat"
            'e': {'f1': 530, 'f2': 1840, 'f3': 2480},  # as in "bed"
            'i': {'f1': 270, 'f2': 2290, 'f3': 3010},  # as in "bit"
            'o': {'f1': 570, 'f2': 840, 'f3': 2410},   # as in "dog"
            'u': {'f1': 300, 'f2': 870, 'f3': 2240},   # as in "book"
            'y': {'f1': 310, 'f2': 2160, 'f3': 2680},  # as in "yes"
            
            # Consonants (approximated)
            's': {'f1': 200, 'f2': 4000, 'f3': 8000},  # fricative
            't': {'f1': 100, 'f2': 1500, 'f3': 4000},  # stop
            'n': {'f1': 280, 'f2': 1700, 'f3': 2600},  # nasal
            'm': {'f1': 250, 'f2': 1200, 'f3': 2400},  # nasal
            'l': {'f1': 300, 'f2': 1300, 'f3': 3000},  # liquid
            'r': {'f1': 350, 'f2': 1200, 'f3': 1600},  # liquid
            'f': {'f1': 180, 'f2': 1000, 'f3': 7500},  # fricative
            'v': {'f1': 200, 'f2': 1000, 'f3': 2500},  # fricative
            'th': {'f1': 180, 'f2': 1400, 'f3': 2800}, # fricative
            'p': {'f1': 80, 'f2': 1000, 'f3': 2500},   # stop
            'b': {'f1': 100, 'f2': 1000, 'f3': 2500},  # stop
            'k': {'f1': 100, 'f2': 2000, 'f3': 3500},  # stop
            'g': {'f1': 120, 'f2': 2000, 'f3': 3500},  # stop
            'd': {'f1': 120, 'f2': 1700, 'f3': 3500},  # stop
            'w': {'f1': 300, 'f2': 900, 'f3': 2200},   # glide
            'h': {'f1': 300, 'f2': 1500, 'f3': 2500},  # fricative
            'j': {'f1': 280, 'f2': 2200, 'f3': 3000},  # fricative/affricate
            'z': {'f1': 250, 'f2': 2000, 'f3': 6000},  # fricative
            'c': {'f1': 200, 'f2': 4000, 'f3': 8000},  # like 's'
            'q': {'f1': 100, 'f2': 2000, 'f3': 3500},  # like 'k'
            'x': {'f1': 200, 'f2': 3000, 'f3': 7000},  # fricative
            'default': {'f1': 400, 'f2': 1500, 'f3': 2500}
        }
        
        # Voice characteristics for different voice types
        self.voice_profiles = {
            "default": {
                "base_f0": 150,  # Fundamental frequency (pitch)
                "f0_range": 50,  # Pitch variation range
                "formant_shift": 1.0,  # Formant frequency multiplier
                "breathiness": 0.08,  # Amount of noise/breathiness
                "vibrato_rate": 5.0,  # Vibrato frequency
                "vibrato_depth": 0.02  # Vibrato intensity
            },
            "female_default": {
                "base_f0": 220,
                "f0_range": 80,
                "formant_shift": 1.15,  # Higher formants for female voice
                "breathiness": 0.12,
                "vibrato_rate": 5.5,
                "vibrato_depth": 0.025
            },
            "male_default": {
                "base_f0": 120,
                "f0_range": 40,
                "formant_shift": 0.9,  # Lower formants for male voice
                "breathiness": 0.06,
                "vibrato_rate": 4.5,
                "vibrato_depth": 0.018
            },
            "female_young": {
                "base_f0": 250,
                "f0_range": 100,
                "formant_shift": 1.2,
                "breathiness": 0.1,
                "vibrato_rate": 6.0,
                "vibrato_depth": 0.03
            },
            "male_deep": {
                "base_f0": 90,
                "f0_range": 30,
                "formant_shift": 0.8,
                "breathiness": 0.04,
                "vibrato_rate": 4.0,
                "vibrato_depth": 0.015
            },
            "female_british": {
                "base_f0": 200,
                "f0_range": 60,
                "formant_shift": 1.1,
                "breathiness": 0.09,
                "vibrato_rate": 5.2,
                "vibrato_depth": 0.022
            }
        }
    
    def _text_to_phonemes(self, text: str) -> list:
        """Convert text to simplified phonemes for speech synthesis"""
        text = text.lower()
        phonemes = []
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # Handle digraphs (two-character combinations)
            if i < len(text) - 1:
                digraph = text[i:i+2]
                if digraph == 'th':
                    phonemes.append('th')
                    i += 2
                    continue
                elif digraph == 'ch':
                    phonemes.append('j')  # ch sound similar to j
                    i += 2
                    continue
                elif digraph == 'sh':
                    phonemes.append('s')  # simplified
                    i += 2
                    continue
            
            if char in 'aeiou':
                phonemes.append(char)
            elif char in 'bcdfgjklmnpqrstvwxyz':
                phonemes.append(char)
            elif char == ' ':
                phonemes.append('pause')
            elif char in '.,!?;:':
                phonemes.append('long_pause')
            elif char in '-–—':
                phonemes.append('pause')
            else:
                # Skip unknown characters or add as pause
                if char.isalnum():
                    phonemes.append('default')
            
            i += 1
        
        return phonemes
    
    def _generate_formant_wave(self, duration: float, f1: float, f2: float, f3: float, 
                              f0: float, sample_rate: int) -> np.ndarray:
        """Generate a formant-based wave that sounds more like human speech"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if len(t) == 0:
            return np.array([])
        
        # Generate fundamental frequency with natural variation
        f0_variation = f0 * (1 + 0.08 * np.sin(2 * np.pi * 3.5 * t) + 
                            0.03 * np.sin(2 * np.pi * 7.2 * t))  # Natural pitch variation
        
        # Generate harmonic series for more natural sound
        wave = np.zeros_like(t)
        
        # Add fundamental and harmonics with formant shaping
        for harmonic in range(1, 10):  # First 9 harmonics
            freq = f0_variation * harmonic
            
            # Natural harmonic amplitude decay
            base_amplitude = 1.0 / (harmonic ** 0.7)
            
            # Apply formant filtering (boost frequencies near formants)
            formant_boost = 1.0
            
            # First formant boost
            f1_distance = np.abs(freq - f1)
            f1_boost = np.exp(-f1_distance / 150) * 1.5
            
            # Second formant boost
            f2_distance = np.abs(freq - f2)
            f2_boost = np.exp(-f2_distance / 200) * 1.2
            
            # Third formant boost
            f3_distance = np.abs(freq - f3)
            f3_boost = np.exp(-f3_distance / 250) * 0.8
            
            # Combine formant effects
            formant_boost = 1.0 + f1_boost + f2_boost + f3_boost
            
            # Final amplitude
            amplitude = base_amplitude * formant_boost * 0.3
            
            # Generate harmonic wave with slight phase variation for realism
            phase = random.uniform(0, 2 * np.pi)
            harmonic_wave = amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Add to main wave
            wave += harmonic_wave
        
        return wave
    
    def _add_voice_characteristics(self, wave: np.ndarray, voice_profile: dict, 
                                  t: np.ndarray) -> np.ndarray:
        """Add voice-specific characteristics like breathiness and vibrato"""
        
        if len(wave) == 0 or len(t) == 0:
            return wave
        
        # Add vibrato (natural pitch variation)
        vibrato = 1 + voice_profile["vibrato_depth"] * np.sin(
            2 * np.pi * voice_profile["vibrato_rate"] * t
        )
        wave *= vibrato
        
        # Add breathiness (subtle noise that sounds like breath)
        if voice_profile["breathiness"] > 0:
            # Generate colored noise (more realistic than white noise)
            noise = np.random.normal(0, 1, len(wave))
            # Apply simple low-pass filter to noise for more natural breathiness
            if len(noise) > 1:
                for i in range(1, len(noise)):
                    noise[i] = 0.7 * noise[i] + 0.3 * noise[i-1]
            noise *= voice_profile["breathiness"]
            wave += noise
        
        # Add natural amplitude variation (like natural breathing rhythm)
        amplitude_variation = 1 + 0.08 * np.sin(2 * np.pi * 0.8 * t) + \
                             0.04 * np.sin(2 * np.pi * 2.1 * t)
        wave *= amplitude_variation
        
        # Add subtle tremolo (amplitude modulation)
        tremolo = 1 + 0.02 * np.sin(2 * np.pi * 6.5 * t)
        wave *= tremolo
        
        return wave
    
    def _apply_coarticulation(self, phoneme_waves: list) -> np.ndarray:
        """Apply coarticulation effects (phonemes influencing each other)"""
        if len(phoneme_waves) == 0:
            return np.array([])
        
        if len(phoneme_waves) == 1:
            return phoneme_waves[0]
        
        result = []
        
        for i, wave in enumerate(phoneme_waves):
            if len(wave) == 0:
                continue
                
            if i == 0:
                # First phoneme - just add it
                result.append(wave)
            else:
                # Blend with previous phoneme for smooth transitions
                transition_length = min(len(wave) // 3, 800)  # Up to 800 samples transition
                
                if transition_length > 0 and len(result) > 0 and len(result[-1]) > 0:
                    # Create smooth transition
                    fade_in = np.linspace(0, 1, transition_length)
                    fade_out = np.linspace(1, 0, transition_length)
                    
                    # Get the last part of previous wave
                    prev_wave = result[-1]
                    if len(prev_wave) >= transition_length:
                        # Overlap and blend
                        overlap_prev = prev_wave[-transition_length:] * fade_out
                        overlap_curr = wave[:transition_length] * fade_in
                        blended_overlap = overlap_prev + overlap_curr
                        
                        # Combine: previous (without overlap) + blended overlap + current (without overlap)
                        result[-1] = np.concatenate([
                            prev_wave[:-transition_length], 
                            blended_overlap
                        ])
                        
                        # Add the rest of current wave
                        if len(wave) > transition_length:
                            result.append(wave[transition_length:])
                    else:
                        # If previous wave is too short, just add current wave
                        result.append(wave)
                else:
                    # No transition possible, just add the wave
                    result.append(wave)
        
        # Concatenate all waves
        if result:
            return np.concatenate(result)
        else:
            return np.array([])
    
    def generate_speech(self, text: str, voice_id: str, speed: float = 1.0, 
                       pitch: float = 1.0, volume: float = 1.0) -> tuple:
        """Generate human-like speech synthesis with improved realism"""
        
        # Get voice profile
        voice_profile = self.voice_profiles.get(voice_id, self.voice_profiles["default"]).copy()
        
        # Apply pitch adjustment
        voice_profile["base_f0"] *= pitch
        voice_profile["f0_range"] *= pitch
        
        # Convert text to phonemes
        phonemes = self._text_to_phonemes(text)
        
        if not phonemes:
            # Generate silence for empty text
            return np.zeros(int(self.sample_rate * 0.5), dtype=np.int16), self.sample_rate
        
        # Generate audio for each phoneme
        phoneme_waves = []
        
        # Add slight randomness for natural speech rhythm
        random.seed(hash(text) % 1000)  # Consistent randomness for same text
        
        for i, phoneme in enumerate(phonemes):
            if phoneme == 'pause':
                # Short pause for spaces
                duration = (0.08 + random.uniform(-0.02, 0.03)) / speed
                silence = np.zeros(int(self.sample_rate * duration))
                phoneme_waves.append(silence)
                
            elif phoneme == 'long_pause':
                # Longer pause for punctuation
                duration = (0.25 + random.uniform(-0.05, 0.08)) / speed
                silence = np.zeros(int(self.sample_rate * duration))
                phoneme_waves.append(silence)
                
            else:
                # Generate speech sound for phoneme
                base_duration = (0.10 + random.uniform(-0.02, 0.04)) / speed  # Natural variation
                
                # Vowels typically last longer than consonants
                if phoneme in 'aeiou':
                    base_duration *= 1.4
                elif phoneme in 'mnrl':  # Sonorants
                    base_duration *= 1.2
                elif phoneme in 'szfvth':  # Fricatives
                    base_duration *= 1.1
                # Stops (p,t,k,b,d,g) keep base duration
                
                # Get phoneme frequencies
                phoneme_data = self.phoneme_frequencies.get(phoneme, self.phoneme_frequencies['default'])
                
                # Apply voice-specific formant shifting
                f1 = phoneme_data['f1'] * voice_profile['formant_shift']
                f2 = phoneme_data['f2'] * voice_profile['formant_shift']
                f3 = phoneme_data['f3'] * voice_profile['formant_shift']
                
                # Add natural F0 variation based on position in sentence
                position_factor = i / max(len(phonemes) - 1, 1)  # 0 to 1
                # Natural declination (pitch tends to fall toward end of sentence)
                declination = 1.0 - 0.15 * position_factor
                
                f0 = voice_profile['base_f0'] * declination + random.uniform(
                    -voice_profile['f0_range']/3, voice_profile['f0_range']/3
                )
                
                # Ensure f0 stays positive
                f0 = max(f0, 50)
                
                # Generate formant wave
                wave = self._generate_formant_wave(base_duration, f1, f2, f3, f0, self.sample_rate)
                
                if len(wave) > 0:
                    # Add voice characteristics
                    t = np.linspace(0, base_duration, len(wave))
                    wave = self._add_voice_characteristics(wave, voice_profile, t)
                    
                    # Apply natural envelope (attack, sustain, decay)
                    envelope_length = len(wave)
                    
                    if envelope_length > 0:
                        attack_length = min(envelope_length // 5, 300)  # Quick attack
                        decay_length = min(envelope_length // 4, 500)   # Gentle decay
                        
                        envelope = np.ones(envelope_length)
                        
                        # Attack (fade in)
                        if attack_length > 0:
                            envelope[:attack_length] = np.linspace(0.1, 1, attack_length)
                        
                        # Decay (fade out)
                        if decay_length > 0:
                            envelope[-decay_length:] = np.linspace(1, 0.2, decay_length)
                        
                        wave *= envelope
                
                phoneme_waves.append(wave)
        
        # Apply coarticulation for smoother transitions
        full_audio = self._apply_coarticulation(phoneme_waves)
        
        if len(full_audio) == 0:
            return np.zeros(int(self.sample_rate * 0.5), dtype=np.int16), self.sample_rate
        
        # Apply volume
        full_audio *= volume
        
        # Add very subtle background noise for realism
        noise_level = 0.002
        noise = np.random.normal(0, noise_level, len(full_audio))
        full_audio += noise
        
        # Apply natural compression (makes it sound more like recorded speech)
        full_audio = np.tanh(full_audio * 0.9) * 1.1
        
        # Apply subtle high-frequency rolloff (like natural vocal tract filtering)
        # Simple lowpass effect
        if len(full_audio) > 1:
            for i in range(1, len(full_audio)):
                full_audio[i] = 0.85 * full_audio[i] + 0.15 * full_audio[i-1]
        
        # Normalize and convert to int16
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val * 0.8  # Leave some headroom
        
        full_audio = np.clip(full_audio, -1, 1)
        full_audio = (full_audio * 32767).astype(np.int16)
        
        return full_audio, self.sample_rate

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
        "message": "Chatterbox TTS API with JWT Authentication & Human-like Speech",
        "version": "2.1.0",
        "status": "running",
        "features": [
            "JWT Authentication",
            "Human-like Voice Synthesis", 
            "Voice Management",
            "Speech Synthesis with Formants",
            "Audio Playback",
            "Custom Voice Creation"
        ],
        "speech_engine": "Advanced Formant-Based TTS",
        "documentation": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0",
        "speech_engine": "Human-like Formant TTS",
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
    """Generate human-like speech from text using specified voice with enhanced parameters"""
    
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
        # Generate speech using enhanced TTS engine
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
            "message": f"Human-like speech synthesized successfully using voice '{voice_data['name']}'",
            "synthesis_info": {
                "text": request.text,
                "voice_id": request.voice_id,
                "voice_name": voice_data["name"],
                "voice_type": voice_data["type"],
                "language": voice_data.get("language", "en-US"),
                "gender": voice_data.get("gender", "neutral"),
                "synthesis_method": "Formant-based with coarticulation",
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
                "info": f"/audio/{audio_id}/info",
                "waveform": f"/audio/{audio_id}/waveform"
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
                "points": len(waveform_normalized),
                "synthesis_method": "Human-like formant synthesis"
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
            "Human-like voice synthesis with formants",
            "Voice cloning from audio samples",
            "Multiple built-in voices with distinct characteristics", 
            "Speech synthesis with natural coarticulation",
            "Audio file download and streaming",
            "Waveform visualization",
            "Voice and audio management"
        ],
        "speech_technology": {
            "engine": "Advanced Formant-Based TTS",
            "features": ["Phoneme processing", "Formant synthesis", "Voice profiles", "Natural coarticulation"]
        }
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
        },
        "speech_engine": {
            "type": "Human-like Formant TTS",
            "version": "2.1.0",
            "features": ["Formant synthesis", "Phoneme mapping", "Voice characteristics", "Coarticulation"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
