import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medicalindex"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def get_retriever():
    """
    Get a retriever from the Pinecone vectorstore.
    Use this in your LLM chain.
    
    Returns:
        Pinecone retriever for RAG queries
    """
    
    embed_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    vectorstore = PineconeVectorStore(index=index, embedding=embed_model, text_key="page_content")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    logger.info("✅ Retriever initialized")
    
    return retriever


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Loading Medical Vectorstore Retriever...")
    logger.info("=" * 50)
    retriever = get_retriever()
    logger.info("✅ Retriever ready for use")