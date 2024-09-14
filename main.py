from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
import whisper
import librosa
import soundfile as sf
from gtts import gTTS
from playsound import playsound
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
import csv

# Initialize Flask application
app = Flask(__name__)

# Load necessary NLTK data
nltk.download('punkt')

# Global variable to store the last generated answer
last_answer = None

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

# Load the Sentence Transformer model
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Load a generative language model for open-ended answers (using T5)
generative_model = pipeline('text2text-generation', model='t5-base')

# Function to split text into sentences
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function to create overlapping paragraphs with more sentences per chunk
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

# Check if the embeddings and index already exist
if os.path.exists(embeddings_file) and os.path.exists(index_file):
    # Load paragraph embeddings from disk
    paragraph_embeddings = np.load(embeddings_file)
    
    # Load the FAISS index directly without passing an existing index object
    paragraph_index = faiss.read_index(index_file)
else:
    # Create embeddings for each paragraph
    paragraph_embeddings = embedding_model.encode(paragraphs)
    
    # Save paragraph embeddings to disk
    np.save(embeddings_file, paragraph_embeddings)
    
    # Initialize a new FAISS index for paragraphs
    dimension = paragraph_embeddings.shape[1]
    paragraph_index = faiss.IndexFlatL2(dimension)
    paragraph_index.add(np.array(paragraph_embeddings))
    
    # Save FAISS index to disk
    faiss.write_index(paragraph_index, index_file)

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

# Initialize Whisper model for Speech-to-Text
stt_model = whisper.load_model("medium")

# Endpoint to handle question asking
@app.route('/ask_question', methods=['POST'])
def ask_question():
    global last_answer  # Use the global variable

    if 'file' in request.files:  # If an audio file is uploaded
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        # Save the uploaded audio file temporarily using a context manager
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        # Transcribe the audio to text
        try:
            transcribed_text = transcribe_audio(temp_file_path)
            
            # Ensure the file is deleted after transcription is done
            os.remove(temp_file_path)

            if not transcribed_text:
                return jsonify({"error": "Failed to transcribe audio."}), 500

            # Get the answer using the QA system
            last_answer = answer_question(transcribed_text)
            return jsonify({"answer": last_answer}), 200

        except Exception as e:
            # Ensure the temporary file is deleted in case of an error
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({"error": str(e)}), 500

    elif 'text' in request.form:  # If a text input is provided
        question = request.form['text']
        if question.strip() == '':
            return jsonify({"error": "Text input is empty."}), 400

        # Get the answer using the QA system
        last_answer = answer_question(question)
        return jsonify({"answer": last_answer}), 200

    else:
        return jsonify({"error": "Invalid input. Provide either 'text' or 'file'."}), 400


# Endpoint to convert text answer to audio and return audio file
@app.route('/get_answer_audio', methods=['GET'])
def get_answer_audio():
    global last_answer  # Use the global variable

    if not last_answer:
        return jsonify({"error": "No answer available. Please use /ask_question first."}), 400

    try:
        # Convert text to speech and save as a temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts = gTTS(text=last_answer, lang='en', tld="co.uk")
        tts.save(temp_audio.name)

        # Return the generated audio file using the updated `send_file` method
        return send_file(temp_audio.name, as_attachment=True, download_name='answer.mp3')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to transcribe audio to text using Whisper model
def transcribe_audio(audio_file_path):
    """
    Transcribes the audio from the provided file path to text using Whisper.
    """
    try:
        # Load audio with librosa
        audio, sr = librosa.load(audio_file_path, sr=16000)
        
        # Save as temporary WAV file with correct format using context manager
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            sf.write(temp_wav.name, audio, sr)
            temp_wav_name = temp_wav.name

        # Transcribe the audio file to text
        result = stt_model.transcribe(temp_wav_name)

        # Ensure the temporary file is deleted after use
        os.remove(temp_wav_name)

        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)