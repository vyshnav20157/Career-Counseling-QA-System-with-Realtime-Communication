# Project Documentation

## Overview

This project has **five** important files: `main.ipynb`, `main.py`, `record_audio.py`, `socketserver.py`, and `socketclient.py`.

- `main.ipynb` focuses on processing text from the given ebook in pdf format, tokenizing the text, and embedding the tokens using the `multi-qa-mpnet-base-dot-v1` model from the `sentence_transformers` library for **indexing** purposes.

- `main.py` creates a Flask Application in which there are **two API endpoints**:
`/ask_question` takes textual or an audio file as input and uses the `T5-Base` model to generate the answer to the user's query. Audio file is processed using Open-AI's `Whisper` module.
`/get_answer_audio` takes the answer received from the prior API call and returns an audio file using `gTTS` that speaks the answer out.

- `record_audio.py` uses `pyaudio` to record an audio file and save it onto the current directory.

- `socketserver.py` creates a server-based **Flask Application with SocketIO** that allows realtime communication with the model via textual or audio input and one can receieve output in text or audio.

- `socketclient.py` is the client application of the `socketserver.py` where you can interact with the server.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Running the Project](#running-the-project)
3. [Explanation of Models used](#explanation-of-models-used)
4. [Conclusion](#conclusion)

## Setup and Installation

### Prerequisites

Ensure you have Python 3.x installed on your machine. The following Python libraries are required:

- `faiss-cpu`: A library for similarity search and clustering.
- `sentence_transformers`: Provides transformer models for generating embeddings.
- `pdfplumber`: Allow parsing of PDF files easily
- `nltk`: Provides tools for text processing such as tokenization.
- `Flask`: A web framework for building web applications in Python.
- `tempfile`: For creating temporary files.
- `whisper`: A library for speech-to-text transcription.
- `librosa`: For audio processing.
- `soundfile`: For reading and writing sound files.
- `gtts`: A library for text-to-speech conversion.
- `numpy`: For numerical operations.
- `transformers`: A library from Hugging Face for state-of-the-art NLP models.
- `csv`: For reading and writing CSV files.
- `pyaudio`: A Python library for audio input and output.
- `wave`: A module in Python's standard library for handling .wav files.
- `Flask-SocketIO`: Adds Socket.IO support to the Flask application, enabling real-time, bidirectional communication between the client and server.
- `socketio`: For client-side Socket.IO functionality to communicate with the server.
- `playsound`: For playing the audio responses received from the server.
- `threading`: For synchronizing the response handling.

### Download NLTK Data

Additionally, you need to download specific NLTK data:

```python
import nltk
nltk.download('punkt')
```

## Running the Project

1. **Data Processing and Embedding Generation(Mainly to view processing of data)**

   Run the Jupyter Notebook that processes the text from a PDF file, tokenizes it, and generates embeddings using the `multi-qa-mpnet-base-dot-v1` model. The code also stores these embeddings in a FAISS index for later retrieval. <br><br>

2. **Main Inference Script**

   This script sets up a Flask web application that provides two main functionalities: a question-answering (QA) system using transformer models and an audio-to-text transcription system using the Whisper model. The application also allows converting the generated answers into audio format.
   Run the file by the following in any CLI: 
   ```
   python main.py
   ```
   This will start the Flask server in debug mode, allowing you to access the endpoints at http://127.0.0.1:5000/ <br>


    ###### To get the answer to a query:

    Send a POST request to /ask_question with either a text input or an audio file like below.
    `curl -X POST -F "text=What is career counseling?" http://127.0.0.1:5000/ask_question`

    `curl -X POST -F "file=@recorded_audio.wav" http://127.0.0.1:5000/ask_question`<br>
    
    ###### To get the answer in audio format:

    Send a GET request to /get_answer_audio like below. Ensure you have asked a question first using `/ask_question`.
    `curl -X GET http://127.0.0.1:5000/get_answer_audio --output answer.mp3`<br><br>

3. **Audio Recording Script**

   Open the application to change the time that the audio is recorded. Default is 5 seconds.
   Run the file by typing the following in any CLI:
   `python Record_Audio.py` <br><br>

4. **Using the Server of SocketIO based Flask Application**

   The socketserver.py script is an extension of the Flask application, which introduces real-time communication using Flask-SocketIO. This allows the application to handle both text and audio inputs from a client, process them, and send responses back in real-time.
   Use the following CLI command to start the server on one terminal
   `python SocketServer.py`
   Wait a while as the server starts and gets hosted.<br><br>

5. **Using the Client of the SocketIO based Flask Application**

   The socketclient.py script serves as a client that communicates with the socketserver.py server using SocketIO. 
   After the server has been hosted, use the following CLI command to start the client on another terminal
   `python SocketClient.py`
   It will bring up a menu to choose between input types. Then choose which option between textual and audio output. Then you will be prompted to write or speak the query as per your input type.

## Explanation of Models used

Here is a brief explanation of the transformer models used in the project and the rationale behind their selection:

1. **`multi-qa-mpnet-base-dot-v1`**: This model, from the `sentence_transformers` library, is designed for generating high-quality sentence embeddings specifically optimized for semantic search and question-answering tasks. This model was chosen because it provides robust and contextually aware embeddings, which are crucial for finding the most relevant paragraphs in a large corpus and enabling efficient retrieval in the FAISS index.

2. **`t5-base`**: This generative transformer model, available via the Hugging Face `transformers` library, is part of the T5 (Text-to-Text Transfer Transformer) family. T5 models convert all NLP problems into a text-to-text format, making them versatile for various tasks like summarization, translation, and open-ended question answering. It was chosen for its capability to produce coherent and human-like text based on input prompts, which is essential for providing meaningful and context-rich responses to user queries.

3. **`Whisper` (medium model)**: Whisper is an automatic speech recognition (ASR) model developed by OpenAI. It is particularly effective in converting speech into text accurately across various accents, languages, and environments. Its strong transcription capabilities ensure that even nuanced spoken queries are correctly interpreted and converted into text for further processing in the QA pipeline.

## Conclusion

This project integrates several powerful tools and libraries to provide a versatile system for question-answering using both text and audio inputs. It leverages advanced transformer models for embedding and text generation, the Whisper model for speech-to-text conversion, and FAISS for efficient similarity search. The supplementary audio recording script adds another layer of interaction, enabling users to record and input audio seamlessly.