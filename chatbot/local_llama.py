from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Query:{question}"),
    ]
)

# streamlit framework

st.title('LangChain Ollama Chatbot')
input_text = st.text_input("Enter your query here")

# Ollama LLM

llm = Ollama(model="gemma3", temperature=0.7)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    # st.write("Please wait for the response...")
    response = chain.invoke({"question": input_text})
    st.write(response)