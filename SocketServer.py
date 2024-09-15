from flask import Flask
from flask_socketio import SocketIO, emit
import os
import tempfile
import whisper
import soundfile as sf
import wave
from gtts import gTTS
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
import csv
import faiss
import re

# Initialize Flask application and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load the Sentence Transformer model
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Load a generative language model for open-ended answers (using T5)
generative_model = pipeline('text2text-generation', model='t5-base')

# Load the tokens from CSV again for reference
def load_tokens_from_csv(filename):
    tokens = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                tokens.append(row[0])
    return tokens

tokens = load_tokens_from_csv("tokens.csv")

# Combine tokens into a single text
full_text = ' '.join(tokens)

# Function to split text into sentences
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function to create overlapping paragraphs with 10 sentences per chunk
def create_overlapping_paragraphs(sentences, chunk_size=10, overlap=4):
    paragraphs = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = sentences[i:i + chunk_size]
        if chunk:  # Ensure the chunk is not empty
            paragraphs.append(' '.join(chunk))
    return paragraphs

# Split the full text into sentences
sentences = split_into_sentences(full_text)

# Create overlapping paragraphs
paragraphs = create_overlapping_paragraphs(sentences, chunk_size=10, overlap=4)

# Define paths for saved embeddings and index
embeddings_file = 'paragraph_embeddings.npy'
index_file = 'paragraph_faiss.index'

# Load paragraph embeddings from disk
paragraph_embeddings = np.load(embeddings_file)

# Load the FAISS index directly without passing an existing index object
paragraph_index = faiss.read_index(index_file)

def get_relevant_paragraphs(question, k=5):
    # Encode the question using the same embedding model
    question_embedding = embedding_model.encode([question])
    
    # Search for the k most similar paragraphs in the FAISS index
    distances, indices = paragraph_index.search(np.array(question_embedding).astype('float32'), k)
    
    # Retrieve the most relevant paragraphs
    relevant_paragraphs = [paragraphs[idx] for idx in indices[0]]
    
    # Combine the relevant paragraphs into a single context for QA
    combined_context = ' '.join(relevant_paragraphs)
    
    return combined_context

def answer_question(question):
    # Get the most relevant context from the FAISS index
    context = get_relevant_paragraphs(question)
    
    # Generate a longer, more informative answer using the generative model
    generated_prompt = f"Based on the following context, provide a detailed and informative answer in 3-5 sentences. Context: {context} Question: {question}"
    generated_answer = generative_model(generated_prompt, max_length=200, min_length=100)
    return generated_answer[0]['generated_text']

def clean_text(text):
    # Remove extra spaces around punctuation
    text = re.sub(r'\s([?.!,:;"](?:\s|$))', r'\1', text)  # No space before punctuation
    text = re.sub(r'([?.!,:;"])(\w)', r'\1 \2', text)  # Ensure space after punctuation

    # Capitalize the first letter of each sentence
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentences
    sentences = [sentence.capitalize() for sentence in sentences]
    text = ' '.join(sentences)
    
    # Correct spacing around quotes and parentheses
    text = re.sub(r'\s+([’])', r'\1', text)  # Remove space before closing quote
    text = re.sub(r'\s+([\'])', r'\1', text)  # Remove space before apostrophes
    text = re.sub(r'([“])\s+', r'\1', text)  # Remove space after opening quote
    text = re.sub(r'\(\s+', '(', text)  # No space after opening parenthesis
    text = re.sub(r'\s+\)', ')', text)  # No space before closing parenthesis
    
    # Fix spacing around dashes or hyphens
    text = re.sub(r'\s*-\s*', ' - ', text)
    
    return text.strip()

# Initialize Whisper model for Speech-to-Text
stt_model = whisper.load_model("medium")

# Function to transcribe audio to text using Whisper model
def transcribe_audio(audio_file_path):
    """
    Transcribes the audio from the provided file path to text using Whisper.
    """
    try:
        print(f"Loading audio from {audio_file_path}...")
        
        # Using PySoundFile to ensure correct WAV format
        audio, sr = sf.read(audio_file_path)
        print(f"Audio loaded. Sample rate: {sr}, Audio length: {len(audio)} samples")

        # Transcribe the audio file to text
        print("Transcribing audio using Whisper model...")
        result = stt_model.transcribe(audio_file_path)
        print(f"Transcription result: {result}")

        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

# SocketIO event to handle both text and audio inputs
@socketio.on('interact')
def handle_interact(data):
    """
    Handles both text and audio inputs, processes them, and returns the response.
    """
    print(f"Received data: {data}")
    response_format = data.get('response_format', 'text')  # Default to text if not specified
    
    if 'text' in data:
        question = data['text']
        print(f"Received text input: {question}")
        if question.strip() == '':
            emit('response', {'error': 'Text input is empty.'})
        else:
            answer = answer_question(question)
            send_response(answer, response_format)

    elif 'audio' in data:
        audio_data = data['audio']
        print(f"Received audio input of length (in bytes): {len(audio_data)}")

        # Save the received audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            with wave.open(temp_audio.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            temp_audio_path = temp_audio.name
            print(f"Temporary WAV file saved at {temp_audio_path}")

        # Transcribe the audio to text
        try:
            transcribed_text = transcribe_audio(temp_audio_path)
            os.remove(temp_audio_path)  # Clean up the temporary file
            print(f"Transcription completed. Result: {transcribed_text}")

            if not transcribed_text:
                emit('response', {'error': 'Failed to transcribe audio.'})
            else:
                # Get the answer using the QA system
                answer = answer_question(transcribed_text)
                send_response(answer, response_format)

        except Exception as e:
            print(f"Error processing audio input: {e}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            emit('response', {'error': str(e)})

def send_response(answer, response_format):
    """
    Sends the response back to the client in the specified format (text or audio).
    """
    print(answer)
    answer = clean_text(answer)
    print(answer)

    if response_format == 'audio':
        # Convert text to speech and save as a temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts = gTTS(text=answer, lang='en', tld="co.uk")
            tts.save(temp_audio.name)

        # Read the audio file and send it as bytes
        try:
            with open(temp_audio.name, 'rb') as audio_file:
                audio_data = audio_file.read()

            # Emit the audio data back to the client
            emit('response', {'audio': audio_data})
        
        except Exception as e:
            emit('response', {'error': str(e)})
        
        finally:
            # Ensure the temporary audio file is deleted after sending
            if os.path.exists(temp_audio.name):
                try:
                    os.remove(temp_audio.name)
                except Exception as e:
                    print(f"Error deleting temporary file: {e}")
    else:
        # Send the response as text
        emit('response', {'answer': answer})


# Run the Flask-SocketIO app
if __name__ == '__main__':
    socketio.run(app, debug=True)
