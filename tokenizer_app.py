import streamlit as st
import tiktoken

# Page title
st.title("🔤 Tokenizer")
st.write("See how an AI model breaks text into tokens (using tiktoken).")


# Load the dictionary only once (so the app stays fast)
@st.cache_resource
def load_encoding():
    return tiktoken.get_encoding("o200k_base")


load_dictionary = load_encoding()

st.caption(f"Vocabulary size: {load_dictionary.n_vocab:,} tokens")

# Menu (radio buttons instead of the while loop + input())
choice = st.radio(
    "Choose an option:",
    [
        "1. Display token IDs of a sentence",
        "2. Display BOTH token IDs and token pieces of a sentence",
        "3. Display token ID of one token",
    ],
)

# Option 1
if choice.startswith("1"):
    sentence = st.text_input("Enter your statement:")
    if sentence:
        token_id = load_dictionary.encode(sentence)
        st.write("**Token IDs:**", token_id)
        st.write(f"Words: {len(sentence.split())}  |  Tokens: {len(token_id)}")

# Option 2
elif choice.startswith("2"):
    sentence = st.text_input("Enter your statement:")
    if sentence:
        token_id = load_dictionary.encode(sentence)
        rows = []
        for i in range(len(token_id)):
            rows.append(
                {
                    "Token ID": token_id[i],
                    "Token piece": repr(load_dictionary.decode([token_id[i]])),
                }
            )
        st.table(rows)
        st.write(f"Words: {len(sentence.split())}  |  Tokens: {len(token_id)}")

# Option 3
elif choice.startswith("3"):
    token = st.text_input("Enter the token:")
    if token:
        token_id_for_token = load_dictionary.encode(token)
        st.write("**Token ID:**", token_id_for_token)
        if len(token_id_for_token) > 1:
            st.info("This text was split into more than one token.")
