import os
import uuid
import shutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PORT = int(os.getenv("PORT", "8000"))

# Initialize FastAPI
app = FastAPI(
    title="Chatterbox TTS API - Memory Optimized",
    description="Text-to-Speech API with lazy-loaded Chatterbox TTS",
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

# Storage directories
VOICES_DIR = Path("voices")
AUDIO_DIR = Path("audio")
VOICES_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# In-memory storage
voices_db = {}
audio_db = {}

# Global TTS model - loaded only when needed
_tts_model = None
_model_loading = False

def get_tts_model():
    """Lazy load the TTS model only when needed"""
    global _tts_model, _model_loading
    
    if _tts_model is not None:
        return _tts_model
    
    if _model_loading:
        raise Exception("Model is currently loading, please wait...")
    
    try:
        _model_loading = True
        logger.info("🚀 Loading Chatterbox TTS model on-demand...")
        
        # Import only when needed
        from chatterbox.tts import ChatterboxTTS
        
        # Force garbage collection before loading
        gc.collect()
        
        # Use CPU and optimize memory
        device = "cpu"
        logger.info(f"Loading model on device: {device}")
        
        # Load model
        _tts_model = ChatterboxTTS.from_pretrained(device=device)
        
        # Force garbage collection after loading
        gc.collect()
        
        logger.info("✅ Chatterbox TTS model loaded successfully")
        return _tts_model
        
    except Exception as e:
        logger.error(f"Failed to load Chatterbox TTS: {e}")
        raise Exception(f"TTS model loading failed: {str(e)}")
    finally:
        _model_loading = False

def generate_speech(
    text: str, 
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    audio_prompt_path: Optional[str] = None
) -> tuple[np.ndarray, int]:
    """Generate speech using Chatterbox TTS with memory management"""
    
    # Get model (loads on first use)
    model = get_tts_model()
    
    try:
        # Clean text
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Limit text length for memory efficiency
        if len(text) > 300:
            text = text[:300]
            logger.warning("Text truncated to 300 characters for memory efficiency")
        
        logger.info(f"Generating speech: '{text[:30]}...'")
        
        # Force garbage collection before generation
        gc.collect()
        
        # Generate speech
        if audio_prompt_path and Path(audio_prompt_path).exists():
            wav = model.generate(
                text, 
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        else:
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        
        # Convert to numpy and clean up
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        
        if wav.ndim > 1:
            wav = wav.squeeze()
        
        # Get sample rate
        sample_rate = getattr(model, 'sr', 24000)
        
        # Force garbage collection after generation
        gc.collect()
        
        logger.info(f"✅ Speech generated: {len(wav)} samples at {sample_rate}Hz")
        return wav, sample_rate
        
    except Exception as e:
        # Clean up on error
        gc.collect()
        logger.error(f"Speech generation failed: {e}")
        raise Exception(f"TTS generation failed: {str(e)}")

# Pydantic models
class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=300)  # Reduced for memory efficiency
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.8, ge=0.05, le=5.0)

def get_audio_duration(file_path: str) -> float:
    try:
        data, samplerate = sf.read(file_path)
        return len(data) / samplerate
    except:
        return 0.0

# Routes
@app.get("/")
async def root():
    return {
        "message": "Chatterbox TTS API - Memory Optimized for Render Free Tier",
        "version": "1.0.0",
        "status": "running",
        "model_status": "loaded" if _tts_model is not None else "will load on first use",
        "note": "Model loads on first synthesis request to save memory"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": _tts_model is not None,
        "model_loading": _model_loading,
        "memory_note": "Model loads on-demand to optimize memory usage"
    }

@app.post("/load-model")
async def load_model():
    """Manually load the TTS model"""
    try:
        model = get_tts_model()
        return {
            "success": True,
            "message": "Model loaded successfully",
            "model_loaded": True
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to load model: {str(e)}",
            "model_loaded": False
        }

@app.post("/synthesize")
async def synthesize_speech(request: SynthesizeRequest):
    try:
        logger.info(f"Synthesis request: {request.text[:50]}...")
        
        # Generate speech (model loads automatically if needed)
        wav, sample_rate = generate_speech(
            text=request.text,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature
        )
        
        # Save audio file
        audio_id = f"audio_{uuid.uuid4().hex}"
        audio_path = AUDIO_DIR / f"{audio_id}.wav"
        
        sf.write(str(audio_path), wav, sample_rate)
        
        # Store metadata
        duration = len(wav) / sample_rate
        audio_data = {
            "audio_id": audio_id,
            "text": request.text,
            "file_path": str(audio_path),
            "sample_rate": sample_rate,
            "duration": duration,
            "created_at": datetime.utcnow().isoformat()
        }
        
        audio_db[audio_id] = audio_data
        
        # Clean up memory
        del wav
        gc.collect()
        
        return {
            "success": True,
            "audio_id": audio_id,
            "message": "Speech synthesized successfully",
            "sample_rate": sample_rate,
            "duration": duration,
            "download_url": f"/audio/{audio_id}"
        }
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        # Clean up on error
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.get("/audio/{audio_id}")
async def download_audio(audio_id: str):
    if audio_id not in audio_db:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio = audio_db[audio_id]
    file_path = audio["file_path"]
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"{audio_id}.wav"
    )

@app.get("/audio/{audio_id}/info")
async def get_audio_info(audio_id: str):
    if audio_id not in audio_db:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio = audio_db[audio_id]
    audio_info = audio.copy()
    audio_info.pop("file_path", None)
    return audio_info

@app.delete("/clear-cache")
async def clear_cache():
    """Clear audio cache to free memory"""
    try:
        # Clear audio files
        for audio_id, audio_data in list(audio_db.items()):
            file_path = Path(audio_data["file_path"])
            if file_path.exists():
                file_path.unlink()
        
        # Clear database
        audio_db.clear()
        
        # Force garbage collection
        gc.collect()
        
        return {
            "success": True,
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to clear cache: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        workers=1  # Use only 1 worker to minimize memory usage
    )
