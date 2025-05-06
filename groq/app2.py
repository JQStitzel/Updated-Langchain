# this is from the 9th video in the Langchain series - he uses app.py throughout, but we already have that

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

## load the keys

groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Groq with Llama 3")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provide context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """)

doc_prompt = st.text_input("Enter your question from documents")

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader("./groq/us_census") ## data ingestion
        st.session_state.documents = st.session_state.loader.load() ## document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) ## chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents[:20]) ## splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) ## vector embeddings

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector store DB is ready")

import time

if doc_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': doc_prompt})
    print("Response time: ", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document similarity search results"):
        # Find and write the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------")