import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
import csv
import os

# Load necessary NLTK data
nltk.download('punkt')

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

# Example usage:
question = "What are good careers in tech?"
answer = answer_question(question)
print(f"Question: {question}\nAnswer: {answer}")