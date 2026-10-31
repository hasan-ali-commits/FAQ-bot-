import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page title
st.title("🤖 FAQ Bot")
st.write("Ask me a question about AI, ML, Deep Learning or Python.")


# Load the model only once (so the app stays fast)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()

# FAQs
faq_questions = [
    "What is AI?",
    "What is Machine Learning?",
    "How does Deep Learning work?",
    "What is Python used for?",
]

faq_answers = [
    "AI enables machines to mimic human intelligence.",
    "Machine Learning allows systems to learn from data.",
    "Deep Learning uses neural networks with many layers.",
    "Python is widely used in AI, ML, and web development.",
]


# Make FAQ embeddings only once
@st.cache_data
def get_faq_embeddings(questions):
    return model.encode(questions)


faq_embeddings = get_faq_embeddings(faq_questions)

# User query (text box instead of input())
query = st.text_input("Ask your question:", placeholder="e.g. What is DL?")

if query:
    # Embedding for the user's question
    query_embedding = model.encode([query])

    # Similarity
    scores = cosine_similarity(query_embedding, faq_embeddings)

    # Best answer
    best_index = np.argmax(scores)
    best_score = scores[0][best_index]

    st.subheader("Most Relevant FAQ:")
    st.write(faq_questions[best_index])

    st.subheader("Answer:")
    st.success(faq_answers[best_index])

    st.caption(f"Similarity score: {best_score:.2f}")