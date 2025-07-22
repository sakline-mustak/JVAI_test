import os
import requests
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from dotenv import load_dotenv
import io
import time
from requests_toolbelt.multipart.encoder import MultipartEncoder
from pydub import AudioSegment
import tempfile
import subprocess


# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice
SAMPLE_RATE = 44100
RECORD_SECONDS = 5  # Max recording duration
DEBUG_MODE = True  # Set to False in production

def record_audio():
    """Record audio from microphone with error handling"""
    try:
        print("\nRecording... Speak now! (Ctrl+C to stop early)")
        recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                          samplerate=SAMPLE_RATE,
                          channels=1,
                          dtype='int16')
        sd.wait()
        print("Recording complete.")
        
        # Convert to WAV in memory
        buffer = io.BytesIO()
        write(buffer, SAMPLE_RATE, recording)
        buffer.seek(0)
        
        if DEBUG_MODE:
            debug_filename = f"debug_recording_{int(time.time())}.wav"
            with open(debug_filename, 'wb') as f:
                f.write(buffer.getvalue())
            print(f"Debug: Audio saved to {debug_filename}")
            
        return buffer
        
    except Exception as e:
        raise Exception(f"Recording failed: {str(e)}")

def transcribe_with_elevenlabs(audio_buffer):
    """Transcribe audio using ElevenLabs STT with robust error handling"""
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    try:
        # Use correct model ID: scribe_v1
        mp_encoder = MultipartEncoder(
            fields={
                'model_id': 'scribe_v1',
                'file': ('recording.wav', audio_buffer.getvalue(), 'audio/wav')
            }
        )
        
        headers['Content-Type'] = mp_encoder.content_type
        
        if DEBUG_MODE:
            print("Sending to ElevenLabs STT...")
        
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            data=mp_encoder,
            headers=headers,
            timeout=30
        )
        
        if DEBUG_MODE:
            print(f"STT Response: {response.status_code}")
        
        if response.status_code == 200:
            return response.json().get("text", "").strip()
        else:
            error_msg = f"STT Error {response.status_code}"
            if response.text:
                error_msg += f": {response.text}"
            raise Exception(error_msg)
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error during STT: {str(e)}")
    except Exception as e:
        raise Exception(f"STT processing failed: {str(e)}")

def get_llama_response(prompt):
    """Get response from Groq's LLaMA with error handling"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        if DEBUG_MODE:
            print(f"Sending to Groq: {prompt[:50]}...")
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if DEBUG_MODE:
            print(f"Groq Response: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error_msg = f"Groq API Error {response.status_code}"
            if response.text:
                error_msg += f": {response.text}"
            raise Exception(error_msg)
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error during Groq request: {str(e)}")
    except Exception as e:
        raise Exception(f"AI processing failed: {str(e)}")

def text_to_speech(text):
    """Convert text to speech using ElevenLabs TTS with ffplay playback"""
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    try:
        if DEBUG_MODE:
            print(f"Converting to speech: {text[:50]}...")

        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers=headers,
            json=payload,
            timeout=30
        )

        if DEBUG_MODE:
            print(f"TTS Response: {response.status_code}")

        if response.status_code == 200:
            # Use pydub to load mp3 from response bytes
            audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                audio.export(temp_file.name, format="mp3")
                subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_file.name],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        else:
            error_msg = f"TTS Error {response.status_code}"
            if response.text:
                error_msg += f": {response.text}"
            raise Exception(error_msg)

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error during TTS: {str(e)}")
    except Exception as e:
        raise Exception(f"TTS processing failed: {str(e)}")


def main():
    print("\n=== Voice AI Assistant ===")
    print("One-shot voice interaction with LLaMA 3.1\n")
    
    try:
        # Step 1: Record audio
        audio_buffer = record_audio()
        
        # Step 2: Transcribe
        user_text = transcribe_with_elevenlabs(audio_buffer)
        if not user_text:
            print("No speech detected in recording.")
            return
            
        print(f"\nYou said: {user_text}")
        
        # Step 3: Get AI response
        print("\nProcessing with LLaMA 3.1...")
        ai_response = get_llama_response(user_text)
        print(f"\nAI Response: {ai_response}")
        
        # Step 4: Convert to speech
        print("\nConverting response to speech...")
        text_to_speech(ai_response)
        
        print("\nInteraction complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
