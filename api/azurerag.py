from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
from fastapi import Request
from openai import AzureOpenAI
from datetime import datetime

load_dotenv()

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A simple API server"
)

# PDF loader
attention_loader = PyPDFLoader(os.path.join(os.path.dirname(__file__), '../rag/attention.pdf'))
attention_docs = attention_loader.load()

user_guide_loader = PyPDFLoader(os.path.join(os.path.dirname(__file__), '../rag/genesis_user_guide.pdf'))
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

try:
    combo_db = FAISS.load_local("storage/combo_index", embedding, allow_dangerous_deserialization=True)
except Exception as e:
    combo_db = FAISS.from_documents(all_documents, embedding)    
    combo_db.save_local("storage/combo_index")

print('done')

db = combo_db



## Load Ollama model
#llm = OllamaLLM(model="gemma3")


endpoint = os.getenv("NATL_AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("NATL_AZURE_OPENAI_MODEL_NAME")
deployment = os.getenv("NATL_AZURE_OPENAI_MODEL__DEPLOYMENT_NAME")

subscription_key = os.getenv("NATL_AZURE_OPENAI_KEY")
api_version = "2024-12-01-preview"

llm = AzureChatOpenAI(
    azure_deployment=deployment,
    openai_api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    temperature=0.7
)



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

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

retriever = db.as_retriever()



retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.post("/query")
async def query(request: Request):
    """
    Endpoint to handle user queries and return responses using the retrieval chain.
    """
    data = await request.json()
    user_input = data.get("input", "")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time}: {user_input}")
    
    if not user_input:
        return {"error": "No input provided"}
    
    try:
        response = retrieval_chain.invoke({"input": user_input})

        print(response['answer'])

        return {"answer": response['answer']}
    except Exception as e:
        return {"error": str(e)}


# user_prompt = ""

# while (user_prompt != 'stop'):

#     user_prompt = input("Enter your question (type 'stop' to exit): ")

#     if (user_prompt != 'stop'):

#         print('processing...')

#         response = retrieval_chain.invoke({"input": user_prompt})

#         print(response['answer'])


if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)