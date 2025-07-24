#!/usr/bin/env python3
"""
Test client for Chatterbox TTS API with JWT authentication
"""

import requests
import json
import time
from typing import Optional

class ChatterboxClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.access_token = None
        self.refresh_token = None
        self.session = requests.Session()
    
    def _get_headers(self):
        """Get headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers
    
    def register(self, username: str, email: str, password: str, full_name: str = None):
        """Register a new user"""
        data = {
            "username": username,
            "email": email,
            "password": password,
            "full_name": full_name
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/register",
            json=data
        )
        
        if response.status_code == 200:
            print(f"✅ User {username} registered successfully")
            return response.json()
        else:
            print(f"❌ Registration failed: {response.text}")
            return None
    
    def login(self, username: str, password: str):
        """Login and get JWT tokens"""
        data = {
            "username": username,
            "password": password
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json=data
        )
        
        if response.status_code == 200:
            tokens = response.json()
            self.access_token = tokens["access_token"]
            self.refresh_token = tokens["refresh_token"]
            print(f"✅ Login successful for {username}")
            return tokens
        else:
            print(f"❌ Login failed: {response.text}")
            return None
    
    def refresh_access_token(self):
        """Refresh the access token"""
        if not self.refresh_token:
            print("❌ No refresh token available")
            return None
        
        response = self.session.post(
            f"{self.base_url}/auth/refresh",
            json={"refresh_token": self.refresh_token}
        )
        
        if response.status_code == 200:
            tokens = response.json()
            self.access_token = tokens["access_token"]
            self.refresh_token = tokens["refresh_token"]
            print("✅ Tokens refreshed successfully")
            return tokens
        else:
            print(f"❌ Token refresh failed: {response.text}")
            return None
    
    def get_profile(self):
        """Get user profile"""
        response = self.session.get(
            f"{self.base_url}/profile",
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            profile = response.json()
            print(f"✅ Profile: {profile}")
            return profile
        else:
            print(f"❌ Failed to get profile: {response.text}")
            return None
    
    def list_voices(self):
        """List available voices"""
        response = self.session.get(
            f"{self.base_url}/voices",
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            voices = response.json()
            print(f"✅ Found {voices['total']} voices ({voices['builtin']} builtin, {voices['custom']} custom)")
            return voices
        else:
            print(f"❌ Failed to list voices: {response.text}")
            return None
    
    def create_voice(self, voice_name: str, voice_description: str, audio_file_path: str):
        """Create a custom voice from audio file"""
        try:
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio_file': audio_file}
                data = {
                    'voice_name': voice_name,
                    'voice_description': voice_description
                }
                
                headers = {}
                if self.access_token:
                    headers["Authorization"] = f"Bearer {self.access_token}"
                
                response = self.session.post(
                    f"{self.base_url}/voices",
                    files=files,
                    data=data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Voice '{voice_name}' created with ID: {result['voice_id']}")
                    return result
                else:
                    print(f"❌ Failed to create voice: {response.text}")
                    return None
        except FileNotFoundError:
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
        except Exception as e:
            print(f"❌ Error creating voice: {str(e)}")
            return None
    
    def synthesize_speech(self, text: str, voice_id: str = "default", speed: float = 1.0, pitch: float = 1.0):
        """Generate speech from text"""
        data = {
            "text": text,
            "voice_id": voice_id,
            "speed": speed,
            "pitch": pitch
        }
        
        response = self.session.post(
            f"{self.base_url}/synthesize",
            json=data,
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Speech synthesized with audio ID: {result['audio_id']}")
            return result
        else:
            print(f"❌ Failed to synthesize speech: {response.text}")
            return None
    
    def download_audio(self, audio_id: str, output_path: str):
        """Download generated audio"""
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        response = self.session.get(
            f"{self.base_url}/audio/{audio_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Audio saved to: {output_path}")
            return True
        else:
            print(f"❌ Failed to download audio: {response.text}")
            return False
    
    def get_audio_info(self, audio_id: str):
        """Get audio metadata"""
        response = self.session.get(
            f"{self.base_url}/audio/{audio_id}/info",
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Audio info: {info}")
            return info
        else:
            print(f"❌ Failed to get audio info: {response.text}")
            return None
    
    def delete_voice(self, voice_id: str):
        """Delete a custom voice"""
        response = self.session.delete(
            f"{self.base_url}/voices/{voice_id}",
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Voice deleted: {result['message']}")
            return result
        else:
            print(f"❌ Failed to delete voice: {response.text}")
            return None

def demo_flow():
    """Demonstrate the complete API workflow"""
    print("🚀 Starting Chatterbox TTS API Demo")
    print("=" * 50)
    
    # Initialize client
    client = ChatterboxClient("http://localhost:8000")  # Change to your Render URL
    
    # Test data
    username = "testuser"
    email = "test@example.com"
    password = "testpassword123"
    
    # 1. Register user
    print("\n1️⃣ Registering user...")
    client.register(username, email, password, "Test User")
    
    # 2. Login
    print("\n2️⃣ Logging in...")
    client.login(username, password)
    
    # 3. Get profile
    print("\n3️⃣ Getting profile...")
    client.get_profile()
    
    # 4. List voices
    print("\n4️⃣ Listing voices...")
    voices = client.list_voices()
    
    # 5. Synthesize speech with default voice
    print("\n5️⃣ Synthesizing speech...")
    text = "Hello! This is a test of the Chatterbox TTS API with JWT authentication."
    result = client.synthesize_speech(text, "default")
    
    if result:
        audio_id = result["audio_id"]
        
        # 6. Get audio info
        print("\n6️⃣ Getting audio info...")
        client.get_audio_info(audio_id)
        
        # 7. Download audio
        print("\n7️⃣ Downloading audio...")
        client.download_audio(audio_id, f"test_output_{audio_id}.wav")
    
    # 8. Test different voices
    print("\n8️⃣ Testing different voices...")
    for voice_type in ["female_default", "male_default"]:
        print(f"\nTesting {voice_type}...")
        result = client.synthesize_speech(
            f"This is a test using the {voice_type} voice.", 
            voice_type
        )
        if result:
            client.download_audio(result["audio_id"], f"{voice_type}_test.wav")
    
    print("\n✅ Demo completed successfully!")
    print("Check the generated audio files in your current directory.")

def interactive_mode():
    """Interactive mode for testing the API"""
    print("🎯 Interactive Chatterbox TTS API Client")
    print("=" * 50)
    
    base_url = input("Enter API base URL (default: http://localhost:8000): ").strip()
    if not base_url:
        base_url = "http://localhost:8000"
    
    client = ChatterboxClient(base_url)
    
    while True:
        print("\nAvailable commands:")
        print("1. Register")
        print("2. Login")
        print("3. Get Profile")
        print("4. List Voices")
        print("5. Create Voice")
        print("6. Synthesize Speech")
        print("7. Download Audio")
        print("8. Delete Voice")
        print("9. Refresh Token")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            username = input("Username: ")
            email = input("Email: ")
            password = input("Password: ")
            full_name = input("Full Name (optional): ") or None
            client.register(username, email, password, full_name)
        elif choice == "2":
            username = input("Username: ")
            password = input("Password: ")
            client.login(username, password)
        elif choice == "3":
            client.get_profile()
        elif choice == "4":
            client.list_voices()
        elif choice == "5":
            name = input("Voice name: ")
            description = input("Voice description: ")
            audio_path = input("Audio file path: ")
            client.create_voice(name, description, audio_path)
        elif choice == "6":
            text = input("Text to synthesize: ")
            voice_id = input("Voice ID (default: default): ") or "default"
            client.synthesize_speech(text, voice_id)
        elif choice == "7":
            audio_id = input("Audio ID: ")
            output_path = input("Output file path: ")
            client.download_audio(audio_id, output_path)
        elif choice == "8":
            voice_id = input("Voice ID to delete: ")
            client.delete_voice(voice_id)
        elif choice == "9":
            client.refresh_access_token()
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        demo_flow()
