RECORD_SECONDS = 5  # Duration to record in seconds, change depending on how long you want to record for

import pyaudio
import wave
import os

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Initialize PyAudio
audio = pyaudio.PyAudio()

def record_audio(output_file_name="recorded_audio.wav"):
    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()

    # Save the recording to a WAV file in the same directory as the code
    with wave.open(output_file_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {output_file_name}")

def main():
    # Record audio and save it in the same directory
    record_audio(output_file_name="recorded_audio.wav")

if __name__ == "__main__":
    main()
