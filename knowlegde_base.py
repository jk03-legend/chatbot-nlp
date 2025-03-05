import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define knowledge base (Q&A)
knowledge_base = {
    "what is AI?": "AI stands for Artificial Intelligence...",
    "who created you?": "I was created by an amazing developer!",
    "how does speech recognition work?": "Speech recognition converts spoken words into text..."
}

# Convert questions to embeddings
questions = list(knowledge_base.keys())
question_vectors = np.array([embedding_model.encode(q) for q in questions]).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(question_vectors.shape[1])
index.add(question_vectors)

# Save index and questions
faiss.write_index(index, "knowledge_base.index")
np.save("knowledge_questions.npy", questions)
