import whisper
import pyaudio
import numpy as np
import librosa
import soundfile as sf
from gtts import gTTS
import os
import tempfile
import wave

# Initialize Whisper model
model = whisper.load_model("large")

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Duration to record in seconds

# Initialize PyAudio
audio = pyaudio.PyAudio()

def record_audio():
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

    # Save the recording to a temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Ensure the file is closed
    temp_wav.close()

    return temp_wav.name

def transcribe_audio(audio_file_path):
    try:
        # Load audio with librosa
        audio, sr = librosa.load(audio_file_path, sr=16000)
        
        # Save as temporary WAV file with correct format
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, audio, sr)
        
        # Ensure the file is closed before processing
        temp_wav.close()

        # Transcribe the audio file to text
        result = model.transcribe(temp_wav.name)
        
        # Clean up the temporary file
        os.remove(temp_wav.name)
        
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def text_to_speech(text, output_file_path):
    # Convert text to speech and save to file
    tts = gTTS(text=text, lang='en', tld="co.uk")
    tts.save(output_file_path)
    print(f"Audio saved to {output_file_path}")

def main():
    # Record audio from the microphone
    audio_file_path = record_audio()

    # Transcribe the recorded audio
    transcribed_text = transcribe_audio(audio_file_path)
    if transcribed_text:
        print(f"Transcribed Text: {transcribed_text}")
        # Convert the transcribed text to speech
        output_speech_file = "output_speech.mp3"
        text_to_speech(transcribed_text, output_speech_file)
    else:
        print("Transcription failed.")

    # Clean up the temporary audio file
    os.remove(audio_file_path)

if __name__ == "__main__":
    main()
