import faiss
import numpy as np
from flask import Flask, request, jsonify
import speech_recognition as sr
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Load model & FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("knowledge_base.index")
knowledge_questions = np.load("knowledge_questions.npy", allow_pickle=True)

# Load predefined answers
knowledge_base = {
    "what is AI?": "AI stands for Artificial Intelligence...",
    "who created you?": "I was created by an amazing developer!",
    "how does speech recognition work?": "Speech recognition converts spoken words into text..."
}

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
        chatbot_response = find_best_match(text)

        return jsonify({"text": text, "chatbot_response": chatbot_response})

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError:
        return jsonify({"error": "Speech recognition service unavailable"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
