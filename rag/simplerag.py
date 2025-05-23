## Data Ingestion
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
import os
from dotenv import load_dotenv

load_dotenv()

# PDF loader
attention_loader = PyPDFLoader('C:/Projects/github/Updated-Langchain/Updated-Langchain/rag/attention.pdf')
attention_docs = attention_loader.load()

user_guide_loader = PyPDFLoader('C:/Projects/github/Updated-Langchain/Updated-Langchain/rag/genesis_user_guide.pdf')
user_guide_docs = user_guide_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
attention_documents = text_splitter.split_documents(attention_docs)
user_guide_documents = text_splitter.split_documents(user_guide_docs)

all_documents = attention_documents + user_guide_documents

print('documents split')

# Vector embedding and vector store
embedding = OllamaEmbeddings(model="nomic-embed-text")


print('starting faiss.from_documents...')

## FAISS Vector Database

## attention
# try:
#     attention_db = FAISS.load_local("storage/attention_index", embedding, allow_dangerous_deserialization=True)
# except Exception as e:
#     attention_db = FAISS.from_documents(attention_documents, embedding)    
#     attention_db.save_local("storage/attention_index")


## user guide
# try:
#     user_guide_db = FAISS.load_local("storage/user_guide_index", embedding, allow_dangerous_deserialization=True)
# except Exception as e:
#     user_guide_db = FAISS.from_documents(user_guide_documents, embedding)    
#     user_guide_db.save_local("storage/user_guide_index")

## combined
try:
    combo_db = FAISS.load_local("storage/combo_index", embedding, allow_dangerous_deserialization=True)
except Exception as e:
    combo_db = FAISS.from_documents(all_documents, embedding)    
    combo_db.save_local("storage/combo_index")

print('done')


# # Load additional PDF
# new_pdf_loader = PyPDFLoader('C:/Projects/github/Updated-Langchain/Updated-Langchain/rag/new_document.pdf')
# new_pdf_docs = new_pdf_loader.load()

# # Split the new document into chunks
# new_pdf_documents = text_splitter.split_documents(new_pdf_docs)

# # Add the new documents to the existing FAISS database
# combo_db.add_documents(new_pdf_documents)

# # Save the updated database
# combo_db.save_local("storage/combo_index")

# print('New documents added and database updated.')


db = combo_db



## Load Ollama model
llm = OllamaLLM(model="gemma3")


## Design chat prompt

prompt = ChatPromptTemplate.from_template("""
                                          Answer the following question based only on the provided context.
                                          Think step by step before providing a detailed answer.
                                          <context>
                                          {context}
                                          </context>
                                          Question: {input}""")

## Chain introduction
## Create stuff documents chain

document_chain = create_stuff_documents_chain(llm,prompt)

retriever = db.as_retriever()



retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = ""

while (prompt != 'stop'):

    prompt = input("Enter your question (type 'stop' to exit): ")

    if (prompt != 'stop'):

        print('processing...')

        response = retrieval_chain.invoke({"input": prompt})

        print(response['answer'])