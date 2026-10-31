import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page title
st.title("📏 Text Similarity Checker")
st.write("Enter two texts to see how similar their meaning is.")


# Load the model only once (so the app stays fast)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()

# Two text boxes instead of input()
text1 = st.text_area("Enter the first text:")
text2 = st.text_area("Enter the second text:")

if st.button("Check similarity"):
    if text1 and text2:
        first_statement = model.encode([text1])
        second_statement = model.encode([text2])

        results = cosine_similarity(first_statement, second_statement)
        results = results[0][0]

        st.metric("Similarity score", f"{results:.2f}")

        if results >= 0.8:
            st.success("Very similar texts")
        elif results >= 0.5:
            st.info("Decent similarity")
        elif results >= 0.3:
            st.warning("Less similarity")
        else:
            st.error("No similarity")
    else:
        st.warning("Please enter both texts.")
