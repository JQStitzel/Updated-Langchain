import requests
import streamlit as st

def get_rag_response(input_text):
    response = requests.post("http://localhost:8000/query",
        json={'input': input_text})
    
    return response.json()['answer']
                                   
def get_poem_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
        json={'input': {'topic': input_text}})
    
    return response.json()["output"]["content"]

st.title('Genesis Chatbot')
st.subheader('Your guide to the Genesis platform')

input_text1 = st.text_input("Ask me about Genesis")
# input_text2 = st.text_input("Poem topic")

if input_text1:
    st.write(get_rag_response(input_text1))

# if input_text2:
#     st.write(get_poem_response(input_text2))

