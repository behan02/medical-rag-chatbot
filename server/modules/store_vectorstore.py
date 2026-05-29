import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Directory for pre-stored medical PDFs
DOCS_DIR = "../docs"
os.makedirs(DOCS_DIR, exist_ok=True)
logger.info(f"PDFs directory: {Path(DOCS_DIR).absolute()}")


# Initialize pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)


def store_vectorstore():
    """
    Load all PDFs from the docs directory into Pinecone vectorstore.
    
    This function:
    1. Finds all PDF files in DOCS_DIR
    2. Splits them into chunks
    3. Generates embeddings
    4. Upserts to Pinecone
    
    Call this once at startup or when adding new PDFs.
    """
    embed_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    
    pdf_files = list(Path(DOCS_DIR).glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"❌ No PDF files found in {DOCS_DIR}")
        logger.info(f"📁 Add your medical PDFs to: {Path(DOCS_DIR).absolute()}")
        return
    
    logger.info(f"📂 Found {len(pdf_files)} PDF files")
    
    for file_path in pdf_files:
        try:
            logger.info(f"📖 Loading: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            logger.info(f"✅ Loaded {len(documents)} pages from {file_path.name}")

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            logger.info(f"✂️  Split into {len(chunks)} chunks")

            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [f"{file_path.stem}-{i}" for i in range(len(chunks))]

            logger.info(f"🔍 Generating embeddings for {len(texts)} chunks...")
        
            # FIX: embed one at a time instead of all at once
            embeddings = []
            for i, text in enumerate(texts):
                embedding = embed_model.embed_query(text)
                embeddings.append(embedding)
                logger.info(f"✅ Embedded chunk {i+1}/{len(texts)}")

            logger.info(f"📦 Number of embeddings returned: {len(embeddings)}")

            # Upsert to Pinecone
            logger.info(f"📤 Upserting to Pinecone index '{PINECONE_INDEX_NAME}'...")
            with tqdm(total=len(embeddings), desc=f"Upserting {file_path.name}") as pbar:
                # Upsert in batches for better performance
                batch_size = 100
                for i in range(0, len(embeddings), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_embeddings = embeddings[i:i+batch_size]
                    batch_texts = texts[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    
                    # Add page_content to metadata for retrieval
                    batch_metadatas_with_content = [
                        {**meta, "page_content": text}
                        for meta, text in zip(batch_metadatas, batch_texts)
                    ]
                    
                    index.upsert(vectors=zip(batch_ids, batch_embeddings, batch_metadatas_with_content))
                    pbar.update(min(batch_size, len(embeddings) - i))

            import time
            time.sleep(2)
            stats = index.describe_index_stats()
            logger.info(f"📊 Total vectors in index after '{file_path.name}': {stats['total_vector_count']}")


            logger.info(f"✅ Successfully indexed {file_path.name}")
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {str(e)}")
            continue
    
    logger.info("🎉 Vectorstore storing complete!")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Storing Medical PDFs to Pinecone...")
    logger.info("=" * 50)
    store_vectorstore()
