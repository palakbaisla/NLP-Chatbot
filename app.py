import streamlit as st
import json
import pickle
import random

# Page configuration
st.set_page_config(page_title="Greetbot", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    .title-text {
        color: white;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle-text {
        color: #e0e0e0;
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .user-message {
        background-color: #667eea;
        color: white;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        border-bottom-left-radius: 4px;
        word-wrap: break-word;
    }
    .bot-message {
        background-color: #764ba2;
        color: white;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        border-bottom-right-radius: 4px;
        word-wrap: break-word;
    }
    .chat-container {
        background-color: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .input-container {
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title-text">🤖 Greetbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Your Friendly AI Greeting Assistant</div>', unsafe_allow_html=True)

try:
    # Load dataset
    with open('intents.json') as file:
        data = json.load(file)
    # Load model and vectorizer
    model = pickle.load(open("model.pkl", 'rb'))
    vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

except Exception as e:
    st.error(f"⚠️ Error loading model or dataset: {e}")
    st.stop()

def chatbot_response(user_input):
    input_vec = vectorizer.transform([user_input.lower()])
    tag = model.predict(input_vec)[0]

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that. Could you rephrase your message?"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat container (display messages first)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if st.session_state.messages:
    # Display chat history
    for sender, message in st.session_state.messages:
        if sender == "You":
            st.markdown(f'<div class="user-message">👤 You: {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🤖 Bot: {message}</div>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color: #999;">Start a conversation...</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-container">', unsafe_allow_html=True)

def handle_input():
    if st.session_state.user_message:
        user_input = st.session_state.user_message
        response = chatbot_response(user_input)
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", response))
        st.session_state.user_message = ""

st.text_input("💬 Type your message here:", placeholder="Ask me anything...", key="user_message", on_change=handle_input)
st.markdown('</div>', unsafe_allow_html=True)