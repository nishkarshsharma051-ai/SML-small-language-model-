import sounddevice as sd
import struct
import math
import threading
import time
import numpy as np

class ClapDetector:
    def __init__(self, callback, threshold=0.03, chunk=1024, rate=44100, required_claps=2):
        self.callback = callback
        self.threshold = threshold
        self.required_claps = required_claps
        self.chunk = chunk
        self.rate = rate
        self.running = False
        self.thread = None

    def get_rms(self, block):
        count = len(block) // 2
        if count == 0:
            return 0.0
        fmt = f"{count}h"
        shorts = struct.unpack(fmt, block)

        sum_squares = 0.0
        for sample in shorts:
            n = sample / 32768.0
            sum_squares += n * n

        return math.sqrt(sum_squares / count)

    def listen(self):
        last_clap_time = 0
        clap_count = 0
        
        try:
            print(f"[ClapDetector] Listening for {self.required_claps} claps using sounddevice...", flush=True)
            with sd.InputStream(samplerate=self.rate, channels=1, dtype='int16') as stream:
                while self.running:
                    data, _ = stream.read(self.chunk)
                    block = data.tobytes()
                    rms = self.get_rms(block)
                    current_time = time.time()
                    
                    if rms > self.threshold:
                        if current_time - last_clap_time > 0.3:
                            clap_count += 1
                            last_clap_time = current_time
                            print(f"[ClapDetector] Clap {clap_count} detected! RMS: {rms:.3f}", flush=True)
                    
                    if clap_count > 0 and current_time - last_clap_time > 1.5:
                        if clap_count == self.required_claps:
                            print(f"[ClapDetector] {self.required_claps} Claps matched! Waking up...", flush=True)
                            if self.callback:
                                self.callback()
                        else:
                            print(f"[ClapDetector] Resetting claps. Heard {clap_count}, needed {self.required_claps}.", flush=True)
                        clap_count = 0
        except Exception as e:
            print(f"[ClapDetector] SoundDevice stream error: {e}", flush=True)
            self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

if __name__ == "__main__":
    print("Starting Clap Detector Test...")
    def on_clap():
        print(">>> CLAP CALLBACK FIRED! <<<")
        
    detector = ClapDetector(callback=on_clap, threshold=0.03)
    detector.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        detector.stop()
        print("Stopped.")
