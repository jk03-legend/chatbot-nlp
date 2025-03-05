from flask import Flask, request, jsonify
import speech_recognition as sr
from flask_cors import CORS
import hashlib
from transformers import pipeline

app = Flask(__name__)
CORS(app)

cache = {}

# Load NLP models
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")  # Example for Q&A
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Knowledge Base (Static)
knowledge_base = {
    "what is your name": "I am an AI-powered assistant.",
    "who created you": "I was created by an amazing developer!",
    "what is ai": "AI stands for Artificial Intelligence, which enables machines to mimic human intelligence."
}

def get_audio_hash(audio_bytes):
    """Generate a stable hash for the audio content."""
    return hashlib.md5(audio_bytes).hexdigest()

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['file']
    audio_bytes = file.read()

    audio_hash = get_audio_hash(audio_bytes)

    if audio_hash in cache:
        return jsonify({"text": cache[audio_hash], "nlp_results": cache[audio_hash + "_nlp"]})

    file.seek(0)  # Reset file pointer
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for noise
            audio = recognizer.record(source, duration=None)  # Capture full audio
        
        results = recognizer.recognize_google(audio, language="en-US", show_all=True)

        if results and "alternative" in results:
            text = results["alternative"][0]["transcript"].lower()  # Normalize text
        else:
            return jsonify({"error": "Could not understand audio"}), 400

        # Check if question is in knowledge base
        answer = knowledge_base.get(text, "I don't know the answer to that. Can you provide more details?")

        # NLP Processing
        sentiment_result = sentiment_analyzer(text)

        nlp_results = {
            "sentiment": sentiment_result,
            "knowledge_base_answer": answer
        }

        cache[audio_hash] = text  # Store recognized text in cache
        cache[audio_hash + "_nlp"] = nlp_results  # Store NLP results in cache

        return jsonify({"text": text, "nlp_results": nlp_results})
    
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError:
        return jsonify({"error": "Speech recognition service unavailable"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
