# importing the required libs here.
from langchain_community.chat_models import ChatOllama
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_groq import ChatGroq

# importing the config file here.
from vsgpt.config import Config

# creating a function to create the LLM.
def create_llm() -> BaseLanguageModel:
    
    # Initiating offline private model from ollama.
    if Config.Model.USE_LOCAL:
        return ChatOllama(
            model= Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMPERATURE,
            keep_alive=f"{Config.Model.KEEP_ALIVE}h",
            max_tokens = Config.Model.MAX_TOKENS
        )
    else:
        # Initializing cloud model from groq.
        return ChatGroq(
            model=Config.Model.REMOTE_LLM,
            temperature=Config.Model.TEMPERATURE,
            max_tokens=Config.Model.MAX_TOKENS
        )
        
# Initiating the vector embeddings model.
def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)

# Initiating the reranker here for the retrieved docs.
def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)

