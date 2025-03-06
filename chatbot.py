import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
from flask import Flask, request, jsonify
import speech_recognition as sr
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Load a lightweight chatbot model
chatbot = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", device="cpu", max_new_tokens=50)

# Knowledge Base (Modify/Add More)
knowledge_base = {
    "what is AI?": "AI stands for Artificial Intelligence...",
    "who created you?": "I was created by an amazing developer!",
    "how does speech recognition work?": "Speech recognition converts spoken words into text..."
}

# Check if FAISS index exists; if not, create it
INDEX_FILE = "knowledge_base.index"
QUESTIONS_FILE = "knowledge_questions.npy"

def create_faiss_index():
    print("Generating FAISS index...")

    # Convert questions into embeddings
    questions = list(knowledge_base.keys())
    question_vectors = np.array([embedding_model.encode(q) for q in questions]).astype("float32")

    # Create FAISS index
    index = faiss.IndexHNSWFlat(question_vectors.shape[1], 32)  # HNSW indexing
    index.hnsw.efSearch = 64

    # Save index & questions
    faiss.write_index(index, INDEX_FILE)
    np.save(QUESTIONS_FILE, questions)

    print("FAISS index created successfully!")

# Ensure index exists before loading
if not os.path.exists(INDEX_FILE) or not os.path.exists(QUESTIONS_FILE):
    create_faiss_index()

# Load FAISS index
index = faiss.read_index(INDEX_FILE)
knowledge_questions = np.load(QUESTIONS_FILE, allow_pickle=True)

def find_best_match(query):
    """Find the closest question in the knowledge base."""
    query_vector = np.array([embedding_model.encode(query)]).astype("float32")
    D, I = index.search(query_vector, 1)  # Find closest match
    best_match = knowledge_questions[I[0][0]] if D[0][0] < 0.5 else None
    return knowledge_base.get(best_match, "I don't have an answer for that.")

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['file']
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.record(source, duration=None)
        
        text = recognizer.recognize_google(audio, language="en-US").lower()

        # Search knowledge base
        if best_match:
            return knowledge_base.get(best_match, "I don't have an answer for that.")
        else:
            chatbot_response = chatbot(query)[0]["generated_text"]
            return chatbot_response


        return jsonify({"chatbot_response": chatbot_response})

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError:
        return jsonify({"error": "Speech recognition service unavailable"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
