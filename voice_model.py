"""
Voice Model for Ting Ling Ling
================================
Provides a stable voice interface for Ting Ling Ling using native macOS synthesis.
Replaces pyttsx3 with direct 'say' command for better reliability in Flask environments.
"""

import subprocess
import time
import os
import sys

# ─── Voice Config ─────────────────────────────────────────────────────────────
# Mappings for the native macOS 'say' command
MALE_VOICES = {
    "daniel":  "Daniel",   # 🎙️ British, clear & professional
    "reed":    "Reed",     # 🎙️ US Eloquence, smooth & natural
    "rocko":   "Rocko",    # 🎙️ US, deep
    "grandpa": "Grandpa",  # 🎙️ US, warm
    "eddy":    "Eddy",     # 🎙️ US Eloquence
    "fred":    "Fred",     # 🎙️ Classic macOS
}

DEFAULT_VOICE = "daniel"
# ─────────────────────────────────────────────────────────────────────────────


class VoiceModel:
    """
    Ting Ling Ling's male voice model.
    Uses macOS 'say' command via subprocess for maximum stability.
    """

    def __init__(self, voice_name: str = DEFAULT_VOICE, rate: int = 160, volume: float = 1.0):
        self.voice_name = voice_name
        self.rate = rate
        self.volume = volume
        self.current_process = None

    def _get_os_voice(self):
        return MALE_VOICES.get(self.voice_name.lower(), MALE_VOICES[DEFAULT_VOICE])

    def speak(self, text: str, label: str = ""):
        """
        Speak the given text aloud using the 'say' command.
        
        Args:
            text: The text to speak.
            label: Optional label printed to the console.
        """
        if label:
            print(f"\n🎙️  [{label}] Speaking with voice: {self.voice_name.capitalize()}...")
        else:
            print(f"\n🎙️  Speaking with voice: {self.voice_name.capitalize()}...")

        # Stop any current speech before starting new one
        self.stop()

        # Clean text for shell (basic sanitization)
        clean_text = text[:1000].replace('"', '').replace("'", "").replace("\n", " ")
        
        os_voice = self._get_os_voice()
        
        # Build command: say -v <Voice> -r <Rate> "Text"
        args = ["say", "-v", os_voice, "-r", str(self.rate), clean_text]
        
        try:
            self.current_process = subprocess.Popen(args)
        except Exception as e:
            print(f"[VoiceModel] Error launching speech process: {e}")

    def stop(self):
        """Stop the current speech process if it's running."""
        if self.current_process and self.current_process.poll() is None:
            try:
                self.current_process.kill()
                self.current_process = None
            except:
                pass

    def set_voice(self, voice_name: str):
        """Switch to a different male voice at runtime."""
        if voice_name.lower() not in MALE_VOICES:
            print(f"[VoiceModel] Unknown voice '{voice_name}'. Available: {list(MALE_VOICES.keys())}")
            return
        self.voice_name = voice_name.lower()

    def list_voices(self):
        """Print all available male voices in the voice model."""
        print("\n🔊 Available Male Voices for Ting Ling Ling:")
        print("─" * 50)
        for name in MALE_VOICES.keys():
            marker = "◀ ACTIVE" if name == self.voice_name.lower() else ""
            print(f"  {name:12} {marker}")
        print("─" * 50)


def speak_text(text: str, voice: str = DEFAULT_VOICE, rate: int = 160):
    """
    Convenience function — speak text without manually creating a VoiceModel.
    Note: This is blocking if you call .wait() but here it follows VoiceModel logic.
    """
    vm = VoiceModel(voice_name=voice, rate=rate)
    vm.speak(text, label="Ting Ling Ling")
    return vm


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform != "darwin":
        print("❌ This voice model requires macOS ('say' command).")
        sys.exit(1)

    print("═" * 55)
    print("   🔊  Ting Ling Ling — Voice Model Demo")
    print("═" * 55)

    vm = VoiceModel(voice_name="daniel", rate=170)
    vm.list_voices()

    demo_text = (
        "Hello! I am Ting Ling Ling, a small language model. "
        "I have been updated to use the native macOS say command for better stability."
    )

    vm.speak(demo_text, label="Intro")
    
    # Give it a moment to start
    time.sleep(2)
    print("\n... Stopping speech early ...")
    vm.stop()

    print("\n✅ Voice model demo complete!")
