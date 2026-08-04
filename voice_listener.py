"""
Voice Listener for Ting Ling Ling
=================================
Records audio from the microphone and uses a local Whisper model to transcribe it to text.
"""

import sounddevice as sd
import numpy as np
import time
from transformers import pipeline

class VoiceListener:
    def __init__(self):
        self.rate = 16000 # Whisper expects 16kHz
        
        print("[VoiceListener] Loading local Whisper model (this may take a moment on first run)...")
        # Load the tiny english whisper model
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
        print("[VoiceListener] Whisper model loaded.")

    def listen(self, duration=5.0):
        """
        Record audio from the microphone for a fixed duration and transcribe it.
        Returns the transcribed text.
        """
        print("[VoiceListener] 🔴 RECORDING NOW...")
        try:
            recording = sd.rec(int(duration * self.rate), samplerate=self.rate, channels=1, dtype='float32')
            sd.wait()
            print("[VoiceListener] ⏹️ Recording finished. Transcribing...")
            audio_data = recording.flatten()
        except Exception as e:
            print(f"[VoiceListener] SoundDevice recording error: {e}")
            return ""

        # Transcribe using the local whisper pipeline
        try:
            result = self.transcriber(audio_data)
            text = result.get("text", "").strip()
            
            # Filter common Whisper silence hallucinations
            lower_text = text.lower()
            if "i'm going to take a look" in lower_text or "thank you." == text or "you" == lower_text:
                return ""
            if len(text) > 100 and len(set(text.split())) < 15:
                # Repetitive hallucination loop
                return ""
                
            print(f"[VoiceListener] 📝 Transcribed: '{text}'")
            return text
        except Exception as e:
            print(f"[VoiceListener] Error during transcription: {e}")
            return ""

    def wait_or_interrupt(self, voice_engine, threshold=0.035, poll_interval=0.1):
        """
        Wait for voice_engine to finish speaking while monitoring the microphone.
        If user speaks (audio RMS exceeds threshold), stop voice_engine immediately.
        Returns True if interrupted, False if completed naturally.
        """
        if not hasattr(voice_engine, "is_speaking") or not voice_engine.is_speaking():
            return False

        chunk_samples = int(poll_interval * self.rate)
        interrupted = False
        
        # Brief warmup pause to avoid initial speaker click self-triggering
        time.sleep(0.4)

        try:
            with sd.InputStream(samplerate=self.rate, channels=1, dtype='float32') as stream:
                while voice_engine.is_speaking():
                    data, overflowed = stream.read(chunk_samples)
                    rms = np.sqrt(np.mean(data**2))
                    if rms > threshold:
                        print(f"[VoiceListener] 🛑 Interruption detected! User voice RMS: {rms:.4f} > threshold ({threshold}). Stopping AI...")
                        voice_engine.stop()
                        interrupted = True
                        break
        except Exception as e:
            print(f"[VoiceListener] Interruption monitor error: {e}")
            if voice_engine.current_process:
                try:
                    voice_engine.current_process.wait()
                except Exception:
                    pass
                
        return interrupted


if __name__ == "__main__":
    print("Testing VoiceListener...")
    listener = VoiceListener()
    print("Test: Talk for 5 seconds...")
    text = listener.listen(duration=5)
    print(f"Result: {text}")
