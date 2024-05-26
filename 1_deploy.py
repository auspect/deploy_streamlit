# %%
# Import necessary modules for the server setup, language processing, and environmental variables.
from fastapi import FastAPI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import os
from langchain.memory import ConversationBufferMemory

import zipfile

# %%
with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
    zip_ref.extractall("chroma_db")

# %%
# Define the model name for text embeddings.
model_name = 'text-embedding-3-small'

# Initialize OpenAIEmbeddings with the specific model, using the retrieved API key.
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv('OPENAI_API_KEY') 
)


# %%

# Set up a vector storage system with specified parameters for embedding functions and persistence.
vectorstore = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# %%

# Create a retriever instance using the previously configured vector store.
retriever = vectorstore.as_retriever()


# %%

# Set up a conversation memory buffer to store chat history, enabling message retrieval.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# %%

# Initialize a retrieval chain that connects language logic models (LLMs) with other components, specifying its type and integrations.
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(), 
    chain_type="map_reduce", 
    retriever=retriever,
    memory=memory  
)


# %%

# Instantiate a FastAPI application with descriptive metadata.
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


# %%
from langserve import add_routes

# Register various application routes that allow interaction with the retriever.
add_routes(app, chain, path="/unique_path")


# %%

# Conditional check for direct script run, initializing server host and port settings.
if __name__ == "__main__":
    import uvicorn  # Import the ASGI server toolkit.

    # Launch the server on the specified host and port.
    uvicorn.run(app, host="localhost", port=8000)


# %%



