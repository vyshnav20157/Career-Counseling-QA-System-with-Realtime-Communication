import socketio
import pyaudio
import wave
import threading

# Flask-SocketIO server URL
SERVER_URL = "http://127.0.0.1:5000"

# Initialize SocketIO client
sio = socketio.Client()

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Duration to record in seconds

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Event object to synchronize between threads
response_event = threading.Event()

def record_audio():
    """
    Records audio from the microphone and returns the recorded audio data.
    """
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

    # Return the recorded audio data as bytes
    audio_data = b''.join(frames)
    print(f"Recorded audio length (in bytes): {len(audio_data)}")
    return audio_data

# Event handler for connecting to the server
@sio.event
def connect():
    print("Connected to the server.")

# Event handler for disconnecting from the server
@sio.event
def disconnect():
    print("Disconnected from the server.")

# Event handler for receiving responses from the server
@sio.on('response')
def on_response(data):
    print("Received response from server.")
    if 'error' in data:
        print(f"Error from server: {data['error']}")
    else:
        if 'answer' in data:
            print(f"Answer from server: {data['answer']}")
        if 'audio' in data:
            with open('response_audio.mp3', 'wb') as f:
                f.write(data['audio'])
            print("Audio response saved as 'response_audio.mp3'.")
    
    # Signal that the response has been received
    response_event.set()

def send_text_query(text):
    """
    Sends a text query to the server via SocketIO.
    """
    # Clear the event before sending a new query
    response_event.clear()
    print(f"Sending text query: {text}")
    sio.emit('interact', {'text': text})

def send_audio_query():
    """
    Records audio and sends it to the server via SocketIO.
    """
    # Clear the event before sending a new query
    response_event.clear()
    audio_data = record_audio()
    print(f"Sending audio query of length (in bytes): {len(audio_data)}")
    sio.emit('interact', {'audio': audio_data})

def main():
    # Connect to the SocketIO server
    print("Connecting to the server...")
    sio.connect(SERVER_URL)

    while True:
        print("\nOptions:")
        print("1. Send Text Query")
        print("2. Send Audio Query")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            text_query = input("Enter your text query: ")
            send_text_query(text_query)
            # Wait for the response to be received
            response_event.wait()
        elif choice == '2':
            print("Recording audio for query...")
            send_audio_query()
            # Wait for the response to be received
            response_event.wait()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

    # Disconnect from the server
    sio.disconnect()

if __name__ == "__main__":
    main()
