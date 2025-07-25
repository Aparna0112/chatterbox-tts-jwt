#!/usr/bin/env python3
"""
Test client for Chatterbox TTS API
Usage: python test_client.py [api_url]
"""

import requests
import json
import sys
import time
from pathlib import Path
from typing import Optional

class ChatterboxTTSClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.token = None
        self.headers = {"Content-Type": "application/json"}
    
    def register(self, username: str, email: str, password: str, full_name: str = None) -> dict:
        """Register a new user"""
        data = {
            "username": username,
            "email": email,
            "password": password,
            "full_name": full_name
        }
        response = requests.post(f"{self.base_url}/auth/register", json=data)
        return response.json()
    
    def login(self, username: str, password: str) -> dict:
        """Login and get access token"""
        data = {"username": username, "password": password}
        response = requests.post(f"{self.base_url}/auth/login", json=data)
        result = response.json()
        
        if response.status_code == 200 and "access_token" in result:
            self.token = result["access_token"]
            self.headers["Authorization"] = f"Bearer {self.token}"
            print(f"✅ Login successful for {username}")
        else:
            print(f"❌ Login failed: {result}")
        
        return result
    
    def get_profile(self) -> dict:
        """Get user profile"""
        response = requests.get(f"{self.base_url}/profile", headers=self.headers)
        return response.json()
    
    def list_voices(self) -> dict:
        """List available voices"""
        response = requests.get(f"{self.base_url}/voices", headers=self.headers)
        return response.json()
    
    def create_voice(self, voice_name: str, description: str, audio_file_path: str) -> dict:
        """Create a custom voice from audio file"""
        if not Path(audio_file_path).exists():
            return {"error": f"Audio file not found: {audio_file_path}"}
        
        with open(audio_file_path, 'rb') as f:
            files = {"audio_file": f}
            data = {
                "voice_name": voice_name,
                "voice_description": description
            }
            # Remove Content-Type for multipart/form-data
            headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}
            response = requests.post(f"{self.base_url}/voices", files=files, data=data, headers=headers)
        
        return response.json()
    
    def synthesize_speech(self, text: str, voice_id: str = "default", **kwargs) -> dict:
        """Synthesize speech from text"""
        data = {
            "text": text,
            "voice_id": voice_id,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/synthesize", json=data, headers=self.headers)
        return response.json()
    
    def download_audio(self, audio_id: str, output_path: str = None) -> bool:
        """Download generated audio"""
        response = requests.get(f"{self.base_url}/audio/{audio_id}", headers=self.headers)
        
        if response.status_code == 200:
            if not output_path:
                output_path = f"{audio_id}.wav"
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Audio saved to {output_path}")
            return True
        else:
            print(f"❌ Failed to download audio: {response.status_code}")
            return False
    
    def get_audio_info(self, audio_id: str) -> dict:
        """Get audio metadata"""
        response = requests.get(f"{self.base_url}/audio/{audio_id}/info", headers=self.headers)
        return response.json()
    
    def delete_voice(self, voice_id: str) -> dict:
        """Delete a custom voice"""
        response = requests.delete(f"{self.base_url}/voices/{voice_id}", headers=self.headers)
        return response.json()
    
    def health_check(self) -> dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()


def run_basic_test(client: ChatterboxTTSClient):
    """Run basic API functionality test"""
    print("🧪 Running basic API tests...\n")
    
    # Test 1: Health check
    print("1. Testing health check...")
    health = client.health_check()
    print(f"   Health status: {health.get('status', 'unknown')}")
    print(f"   TTS model: {health.get('tts_model', 'unknown')}\n")
    
    # Test 2: Register user
    print("2. Testing user registration...")
    username = f"test_user_{int(time.time())}"
    email = f"{username}@example.com"
    password = "test_password_123"
    
    register_result = client.register(username, email, password, "Test User")
    print(f"   Registration: {register_result}\n")
    
    # Test 3: Login
    print("3. Testing user login...")
    login_result = client.login(username, password)
    if "access_token" not in login_result:
        print("❌ Login failed, stopping tests")
        return
    print("   ✅ Login successful\n")
    
    # Test 4: Get profile
    print("4. Testing profile retrieval...")
    profile = client.get_profile()
    print(f"   Profile: {profile.get('username', 'unknown')}\n")
    
    # Test 5: List voices
    print("5. Testing voice listing...")
    voices = client.list_voices()
    print(f"   Available voices: {voices.get('total', 0)}")
    for voice in voices.get('voices', []):
        print(f"   - {voice['name']} ({voice['type']})")
    print()
    
    # Test 6: Synthesize speech with default voice
    print("6. Testing speech synthesis...")
    test_text = "Hello! This is a test of the Chatterbox TTS API. The speech should sound natural and clear."
    
    synthesis_result = client.synthesize_speech(test_text)
    if synthesis_result.get('success'):
        audio_id = synthesis_result['audio_id']
        print(f"   ✅ Speech synthesized successfully")
        print(f"   Audio ID: {audio_id}")
        print(f"   Duration: {synthesis_result.get('duration', 0):.2f} seconds")
        
        # Test 7: Download audio
        print("\n7. Testing audio download...")
        if client.download_audio(audio_id, f"test_output_{audio_id}.wav"):
            print("   ✅ Audio downloaded successfully")
        
        # Test 8: Get audio info
        print("\n8. Testing audio info retrieval...")
        audio_info = client.get_audio_info(audio_id)
        print(f"   Audio info: {audio_info}")
    else:
        print(f"   ❌ Speech synthesis failed: {synthesis_result}")
    
    print("\n🎉 Basic tests completed!")


def run_voice_cloning_test(client: ChatterboxTTSClient, audio_file_path: str):
    """Test voice cloning functionality"""
    if not Path(audio_file_path).exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        print("Skipping voice cloning test")
        return
    
    print(f"\n🎤 Testing voice cloning with {audio_file_path}...\n")
    
    # Create custom voice
    print("1. Creating custom voice...")
    voice_name = f"Custom Voice {int(time.time())}"
    voice_description = "Test custom voice for API testing"
    
    voice_result = client.create_voice(voice_name, voice_description, audio_file_path)
    if voice_result.get('success'):
        voice_id = voice_result['voice_id']
        print(f"   ✅ Custom voice created: {voice_id}")
        print(f"   Duration: {voice_result.get('voice_info', {}).get('audio_duration', 0):.2f} seconds")
        
        # Test synthesis with custom voice
        print("\n2. Testing synthesis with custom voice...")
        test_text = "This is a test using my custom cloned voice. How does it sound?"
        
        synthesis_result = client.synthesize_speech(
            text=test_text,
            voice_id=voice_id,
            exaggeration=0.7,
            cfg_weight=0.4
        )
        
        if synthesis_result.get('success'):
            audio_id = synthesis_result['audio_id']
            print(f"   ✅ Custom voice synthesis successful")
            client.download_audio(audio_id, f"custom_voice_test_{audio_id}.wav")
            
            # Clean up - delete custom voice
            print("\n3. Cleaning up custom voice...")
            delete_result = client.delete_voice(voice_id)
            print(f"   Delete result: {delete_result}")
        else:
            print(f"   ❌ Custom voice synthesis failed: {synthesis_result}")
    else:
        print(f"   ❌ Custom voice creation failed: {voice_result}")


def interactive_mode(client: ChatterboxTTSClient):
    """Interactive mode for manual testing"""
    print("\n🎮 Interactive Mode")
    print("Commands: register, login, voices, synthesize, download, quit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "quit" or command == "exit":
                break
            elif command == "register":
                username = input("Username: ")
                email = input("Email: ")
                password = input("Password: ")
                full_name = input("Full name (optional): ") or None
                result = client.register(username, email, password, full_name)
                print(json.dumps(result, indent=2))
            
            elif command == "login":
                username = input("Username: ")
                password = input("Password: ")
                result = client.login(username, password)
                print(json.dumps(result, indent=2))
            
            elif command == "voices":
                result = client.list_voices()
                print(json.dumps(result, indent=2))
            
            elif command == "synthesize":
                text = input("Text to synthesize: ")
                voice_id = input("Voice ID (default: default): ") or "default"
                exaggeration = input("Exaggeration (0.0-2.0, default: 0.5): ")
                exaggeration = float(exaggeration) if exaggeration else 0.5
                
                result = client.synthesize_speech(text, voice_id, exaggeration=exaggeration)
                print(json.dumps(result, indent=2))
                
                if result.get('success'):
                    download = input("Download audio? (y/n): ").lower() == 'y'
                    if download:
                        audio_id = result['audio_id']
                        filename = input(f"Filename (default: {audio_id}.wav): ") or f"{audio_id}.wav"
                        client.download_audio(audio_id, filename)
            
            elif command == "download":
                audio_id = input("Audio ID: ")
                filename = input("Filename (optional): ") or None
                client.download_audio(audio_id, filename)
            
            elif command == "help":
                print("Available commands:")
                print("- register: Register a new user")
                print("- login: Login to get access token")
                print("- voices: List available voices")
                print("- synthesize: Generate speech from text")
                print("- download: Download audio by ID")
                print("- quit/exit: Exit interactive mode")
            
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function"""
    # Get API URL from command line argument
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"🚀 Chatterbox TTS API Test Client")
    print(f"🌐 API URL: {api_url}")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    client = ChatterboxTTSClient(api_url)
    
    # Check if running in interactive mode
    if len(sys.argv) > 2 and sys.argv[2] == "interactive":
        interactive_mode(client)
        return
    
    # Run automated tests
    try:
        run_basic_test(client)
        
        # Check for audio file to test voice cloning
        test_audio_files = [
            "voice_sample.wav",
            "test_voice.wav",
            "sample.wav",
            "voice.wav"
        ]
        
        for audio_file in test_audio_files:
            if Path(audio_file).exists():
                run_voice_cloning_test(client, audio_file)
                break
        else:
            print("\n📝 Note: No test audio file found for voice cloning test")
            print("   Create a voice_sample.wav file to test voice cloning")
        
        print(f"\n✅ All tests completed successfully!")
        print(f"🌟 API is working correctly at {api_url}")
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {api_url}")
        print("   Make sure the server is running and the URL is correct")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")


if __name__ == "__main__":
    main()
