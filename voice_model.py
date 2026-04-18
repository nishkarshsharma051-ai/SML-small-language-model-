"""
Voice Model for Ting Ling Ling
================================
Provides a male TTS voice to speak any generated text.
Uses macOS native voices via pyttsx3.

Available Male Voices (pre-selected for quality):
  - "daniel"  → Daniel (British English) - clear & professional
  - "reed"    → Reed (US English Eloquence) - smooth & natural
  - "fred"    → Fred (Classic macOS)
  - "grandpa" → Grandpa (Eloquence)
  - "rocko"   → Rocko (Eloquence, deep)
"""

import pyttsx3
import time

# ─── Voice Config ─────────────────────────────────────────────────────────────
MALE_VOICES = {
    "daniel":  "com.apple.voice.compact.en-GB.Daniel",          # 🎙️ British, clear
    "reed":    "com.apple.eloquence.en-US.Reed",                 # 🎙️ US Eloquence, smooth
    "grandpa": "com.apple.eloquence.en-US.Grandpa",             # 🎙️ US, warm
    "rocko":   "com.apple.eloquence.en-US.Rocko",               # 🎙️ US, deep
    "eddy":    "com.apple.eloquence.en-US.Eddy",                # 🎙️ US Eloquence
    "fred":    "com.apple.speech.synthesis.voice.Fred",          # 🎙️ Classic macOS
}

DEFAULT_VOICE = "daniel"   # Best overall quality for Ting Ling Ling
# ─────────────────────────────────────────────────────────────────────────────


class VoiceModel:
    """
    Ting Ling Ling's male voice model.
    Wraps pyttsx3 to provide a clean interface for text-to-speech generation.
    """

    def __init__(self, voice_name: str = DEFAULT_VOICE, rate: int = 160, volume: float = 1.0):
        """
        Initialize the voice model.

        Args:
            voice_name: Name of the voice (e.g., "daniel", "reed", "rocko").
            rate: Speech rate in words-per-minute. Default is 160 (natural pace).
            volume: Volume level 0.0 to 1.0.
        """
        self.engine = pyttsx3.init()
        self.voice_name = voice_name
        self.rate = rate
        self.volume = volume
        self._configure()

    def _configure(self):
        voice_id = MALE_VOICES.get(self.voice_name.lower())
        if voice_id is None:
            print(f"[VoiceModel] Warning: '{self.voice_name}' not found. Using default: '{DEFAULT_VOICE}'")
            voice_id = MALE_VOICES[DEFAULT_VOICE]

        self.engine.setProperty('voice', voice_id)
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)
        print(f"[VoiceModel] Active voice: {self.voice_name.upper()} ({voice_id})")

    def speak(self, text: str, label: str = ""):
        """
        Speak the given text aloud.

        Args:
            text: The text to speak.
            label: Optional label printed to the console.
        """
        if label:
            print(f"\n🎙️  [{label}] Speaking with voice: {self.voice_name.capitalize()}...")
        else:
            print(f"\n🎙️  Speaking with voice: {self.voice_name.capitalize()}...")

        self.engine.say(text)
        self.engine.runAndWait()

    def set_voice(self, voice_name: str):
        """Switch to a different male voice at runtime."""
        if voice_name not in MALE_VOICES:
            print(f"[VoiceModel] Unknown voice '{voice_name}'. Available: {list(MALE_VOICES.keys())}")
            return
        self.voice_name = voice_name
        self._configure()

    def list_voices(self):
        """Print all available male voices in the voice model."""
        print("\n🔊 Available Male Voices for Ting Ling Ling:")
        print("─" * 50)
        for name, vid in MALE_VOICES.items():
            marker = "◀ ACTIVE" if name == self.voice_name else ""
            print(f"  {name:12} | {vid.split('.')[-1]:20} {marker}")
        print("─" * 50)


def speak_text(text: str, voice: str = DEFAULT_VOICE, rate: int = 160):
    """
    Convenience function — speak text without manually creating a VoiceModel.

    Args:
        text: Text to speak.
        voice: Voice name (e.g. 'daniel', 'reed', 'rocko').
        rate: Words-per-minute speech rate.
    """
    vm = VoiceModel(voice_name=voice, rate=rate)
    vm.speak(text, label="Ting Ling Ling")


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 55)
    print("   🔊  Ting Ling Ling — Voice Model Demo")
    print("═" * 55)

    vm = VoiceModel(voice_name="daniel", rate=155)
    vm.list_voices()

    demo_text = (
        "Hello! I am Ting Ling Ling, a small language model. "
        "I have learned to speak with a male voice using the Daniel voice. "
        "Shall I recite some Shakespeare for you?"
    )

    vm.speak(demo_text, label="Intro")
    time.sleep(0.5)

    shakespeare = (
        "To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer "
        "The slings and arrows of outrageous fortune."
    )

    vm.speak(shakespeare, label="Shakespeare")

    print("\n✅ Voice model demo complete!")
