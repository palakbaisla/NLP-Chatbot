import streamlit as st
import json
import pickle
import random #random module is used to predict any random number or any random choice from the given list
st.title("Smart NLP Chatbot")

try:
    #Load dataset
    with open('/Users/palakbaisla/Documents/Anzen_01/NLP/intents.json') as file:
        data=json.load(file)
    #Load model and vectorizer
    model=pickle.load(open("/Users/palakbaisla/Documents/Anzen_01/model.pkl",'rb'))
    vectorizer=pickle.load(open("/Users/palakbaisla/Documents/Anzen_01/vectorizer.pkl",'rb'))

except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

def chatbot_response(user_input):
    input_vec =vectorizer.transform([user_input.lower()])
    tag=model.predict(input_vec)[0]

    for intent in data['intents']:
        if intent['tag']==tag:
            return random.choice(intent['responses'])
        
if "messages" not in st.session_state:
    st.session_state.messages=[]

user_input=st.text_input("You:")

if user_input:
    response = chatbot_response(user_input)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response))

for sender,message in st.session_state.messages:
    if sender=="You":
        st.write(f"You: {message}")
    else:
        st.markdown(f"Bot: {message}")