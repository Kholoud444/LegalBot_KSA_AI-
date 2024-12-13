
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import openai  # Import the OpenAI library
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

# Apply custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #013C59; /* Dark Blue background */
        color: #ffffff; /* White text for contrast */
        font-family: Arial, sans-serif;
    }
    .stApp {
        background-color: #013C59;
    }
    .instruction-text {
        text-align: center;
        font-size: 1.5rem;
        color: #FFA000; /* Orange for instruction text */
        margin-bottom: 20px;
    }
    .credit-text {
        text-align: center;
        font-size: 1rem;
        color: #ffffff; /* White text */
        margin-top: 50px; /* Add spacing above the credit text */
        margin-bottom: 20px;
    }
    .chat-bubble {
        background-color: #004d40; /* Dark green background */
        color: #ffffff; /* White text */
        border-radius: 15px;
        padding: 10px 15px;
        margin: 10px auto;
        max-width: 70%;
        word-wrap: break-word;
    }
    .chat-bubble.user {
        background-color: #FFA000; /* Orange for user message */
        color: #ffffff; /* White text */
    }
    </style>
""", unsafe_allow_html=True)

# Add instruction message above the input box
st.markdown("""
    <div class="instruction-text">
        Your Personal Guide to Labor Law ⚖️<br>Ask and know your rights!
    </div>
""", unsafe_allow_html=True)

# Add "Created by Kholoud" text (visible but not in the footer)
st.markdown("""
    <div class="credit-text">
        Created by <b>Kholoud</b> ❤️ | at Ironhack Barcelona 
    </div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load embeddings and vector store
bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vector_store = FAISS.load_local(
    Path("faiss_index"),
    embeddings=bge_embeddings,
    allow_dangerous_deserialization=True
)

# Function to generate AI responses
def generate_response_with_history(user_input: str, context: str, model: str = "gpt-4o-mini") -> str:
    system_message = f"""
    You are a great AI. You want to help people getting answers to their questions about labor law.
    Please use CONTEXT provided to answer these question. If answers can't be found in the CONTEXT
    tell users so, and advice them to talk to a labor lawyer.

    # CONTEXT:
    {context}
    """
    # Generate response using OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
    ).choices[0].message.content

    return response

# Chat input logic
user_input = st.text_input("Ask me anything:", key="chat_input")

if user_input:
    with st.spinner("Retrieving relevant information..."):
        retrieved_chunks = vector_store.similarity_search(user_input, k=4)

    # Combine chunks for context
    context = "\n".join([chunk.page_content for chunk in retrieved_chunks])
    context = f"""
    Here is some information about smart cities.
    A user will ask you questions. Please answer based on the information provided.

    # INFORMATION
    {context}

    If the INFORMATION doesn't answer the user's question, tell this to the user
    and suggest something based on your knowledge. But WARN THEM.
    """

    # Get response
    response = generate_response_with_history(user_input, context, model="gpt-4o-mini")

    # Add user input and response to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

# Render chat history dynamically
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="chat-bubble user">{message['content']}</div>
        """, unsafe_allow_html=True)
    elif message["role"] == "assistant":
        st.markdown(f"""
            <div class="chat-bubble">{message['content']}</div>
        """, unsafe_allow_html=True)

# Footer (remains fixed at the bottom of the page)
st.markdown("""
    <footer>
        © 2024 | Labor Law Application
    </footer>
""", unsafe_allow_html=True)
