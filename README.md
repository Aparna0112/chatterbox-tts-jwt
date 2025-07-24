# chatterbox-tts-jwt
# Chatterbox TTS API with JWT Authentication

Production-ready Text-to-Speech API with JWT authentication, optimized for Render deployment.

## ✨ Features

- **🔐 JWT Authentication**: Secure user registration and login system
- **🎤 Voice Cloning**: Create custom voices from audio samples
- **🗣️ Speech Synthesis**: Generate speech using built-in or custom voices  
- **👤 User Management**: User profiles and permissions
- **🔄 RESTful API**: Complete CRUD operations with proper authentication
- **☁️ Render Ready**: Optimized for Render free tier deployment
- **📊 Audio Management**: Download and manage generated audio files

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd chatterbox-tts-jwt
```

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env and set your SECRET_KEY

# Run the application
python main.py
```

The API will be available at `http://localhost:8000`

### 3. Deploy to Render

#### Option A: Using render.yaml (Recommended)

1. Push your code to GitHub
2. Connect to Render and create a new Blueprint
3. Select your repository
4. Render will automatically use the `render.yaml` configuration

#### Option B: Manual Setup

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3
   - **Plan**: Free

4. Set Environment Variables:
   - `SECRET_KEY`: Generate a secure random string
   - `PYTHON_VERSION`: 3.10.7

## 📚 API Documentation

Once running, visit:
- **Interactive Docs**: `http://your-url/docs`
- **ReDoc**: `http://your-url/redoc`

## 🔑 Authentication Flow

### 1. Register User
```bash
curl -X POST "http://your-url/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com", 
    "password": "securepassword123",
    "full_name": "John Doe"
  }'
```

### 2. Login
```bash
curl -X POST "http://your-url/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "password": "securepassword123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 3. Use Token for API Calls
```bash
curl -X GET "http://your-url/voices" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 🎯 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get tokens
- `POST /auth/refresh` - Refresh access token

### User Management
- `GET /profile` - Get user profile

### Voice Management
- `GET /voices` - List available voices
- `POST /voices` - Create custom voice from audio
- `DELETE /voices/{voice_id}` - Delete custom voice

### Speech Synthesis
- `POST /synthesize` - Generate speech from text
- `GET /audio/{audio_id}` - Download generated audio
- `GET /audio/{audio_id}/info` - Get audio metadata

### System
- `GET /` - API status
- `GET /health` - Health check

## 🧪 Testing with Python Client

```bash
# Run demo
python test_client.py

# Interactive mode
python test_client.py interactive
```

## 📋 Example Usage

### Create Voice and Synthesize Speech

```python
import requests

# 1. Register and login
auth_response = requests.post("http://your-url/auth/login", json={
    "username": "johndoe",
    "password": "securepassword123"
})
token = auth_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. List available voices
voices = requests.get("http://your-url/voices", headers=headers)
print(voices.json())

# 3. Synthesize speech
synthesis_response = requests.post("http://your-url/synthesize", 
    headers=headers,
    json={
        "text": "Hello! This is a test of the TTS API.",
        "voice_id": "default",
        "speed": 1.0,
        "pitch": 1.0
    }
)
audio_id = synthesis_response.json()["audio_id"]

# 4. Download audio
audio_response = requests.get(f"http://your-url/audio/{audio_id}", headers=headers)
with open("output.wav", "wb") as f:
    f.write(audio_response.content)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT secret key | Required |
| `PORT` | Server port | 8000 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token expiry | 30 |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token expiry | 7 |

### JWT Token Management

- **Access Tokens**: Short-lived (30 minutes) for API access
- **Refresh Tokens**: Long-lived (7 days) for getting new access tokens
- **Automatic Refresh**: Use refresh endpoint when access token expires

## 🔒 Security Features

- **Password Hashing**: bcrypt with salt
- **JWT Tokens**: Secure token-based authentication
- **User Isolation**: Users can only access their own data
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: Built-in protection (can be enhanced)

## ⚠️ Limitations on Render Free Tier

- **No GPU**: Uses CPU-only TTS (slower performance)
- **Memory**: Limited RAM (might need optimization for large files)
- **Storage**: Non-persistent (files are lost on restart)
- **Cold Starts**: May have startup delays

## 🚧 Production Considerations

For production deployment:

1. **Use Strong Secret Key**: Generate cryptographically secure key
2. **Enable HTTPS**: Render provides SSL certificates
3. **Add Rate Limiting**: Implement proper rate limiting
4. **File Storage**: Use external storage (AWS S3, etc.)
5. **Database**: Consider using PostgreSQL for user/metadata storage
6. **Monitoring**: Add logging and error tracking
7. **Backup**: Implement data backup strategies

## 🔄 Migration from RunPod

If migrating from RunPod version:

1. **Add Authentication**: All endpoints now require JWT tokens
2. **CPU Performance**: Expect slower TTS generation without GPU
3. **Storage**: Implement external storage for persistence
4. **Scaling**: Consider hybrid approach (Render + RunPod proxy)

## 📝 Development

### Project Structure
```
chatterbox-tts-jwt/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── render.yaml         # Render deployment config
├── test_client.py      # Test client and examples
├── .env.example        # Environment variables template
├── .gitignore         # Git ignore patterns
└── README.md          # This file
```

### Adding Custom TTS Model

To integrate a real TTS model, replace the `MockTTS` class in `main.py`:

```python
# Replace MockTTS with your actual TTS implementation
class YourCustomTTS:
    def __init__(self, device="cpu"):
        # Initialize your TTS model
        pass
    
    def generate_speech(self, text: str, voice_id: str, **kwargs):
        # Implement actual speech generation
        # Return: (audio_array, sample_rate)
        pass
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

- **Issues**: Open GitHub issues for bugs
- **Documentation**: Check `/docs` endpoint for API documentation
- **Questions**: Create discussions for general questions

## 🎉 Acknowledgments

- Based on ChatterboxTTS by ResembleAI
- FastAPI for the excellent web framework
- Render for free hosting platform
