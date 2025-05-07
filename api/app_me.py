from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langserve import add_routes
from langchain_community.llms import Ollama
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A simple API server"
)

add_routes(
    app,
    ChatOpenAI(),
    path = "/openai"
)

model = ChatOpenAI()
llm = ChatOllama(model="gemma3")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with between 80 and 100 words. Reply with just the essay and nothing else.")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with between 80 and 100 words. Reply with just the poem and nothing else.")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)

add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)