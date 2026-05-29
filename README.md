# 🩺 Medical RAG Chatbot

An AI-powered medical chatbot that leverages **Retrieval-Augmented Generation (RAG)** to answer medical questions based on uploaded documents. Built with FastAPI, LangChain, Pinecone, and Streamlit.

## Features

- 📄 **Document-based QA**: Upload medical PDFs and get answers based on document content
- 💬 **Chat History**: Maintains conversation context for follow-up questions
- 🔍 **Vector Search**: Uses Pinecone for semantic similarity search
- 🤖 **LLM Integration**: Powered by Llama 3.3 via Groq API
- 🎯 **Accurate Embeddings**: Google's Generative AI embeddings for semantic understanding

## Architecture

```
medical-rag-chatbot/
├── client/                 # Streamlit frontend
│   ├── app.py             # Main Streamlit app
│   ├── config.py          # API configuration
│   ├── requirements.txt    # Client dependencies
│   ├── components/
│   │   └── chatUI.py      # Chat interface component
│   └── utils/
│       └── api.py         # API communication
│
├── server/                # FastAPI backend
│   ├── main.py            # FastAPI app setup
│   ├── logger.py          # Logging configuration
│   ├── requirements.txt    # Server dependencies
│   ├── middlewares/
│   │   └── exception_handlers.py
│   ├── modules/
│   │   ├── llm.py                # LLM chain setup
│   │   ├── load_vectorstore.py   # Retriever initialization
│   │   └── store_vectorstore.py  # Document ingestion
│   └── routes/
│       └── ask_question.py       # Chat endpoint
│
└── docs/                  # Directory for PDF documents
```

## Prerequisites

- Python 3.13+
- API Keys:
  - [Groq API Key](https://console.groq.com) (for LLM)
  - [Google API Key](https://cloud.google.com) (for embeddings)
  - [Pinecone API Key](https://www.pinecone.io) (for vector database)

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd medical-rag-chatbot

# Install server dependencies
cd server
pip install -r requirements.txt

# Install client dependencies (in a new terminal)
cd client
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 3. Add Medical Documents

Place your medical PDF files in the `docs/` directory:

```bash
mkdir -p docs
# Add your .pdf files to the docs/ folder
```

### 4. Initialize Vector Store

Run this once to process PDFs and populate Pinecone:

```bash
cd server
python modules/store_vectorstore.py
```

### 5. Run the Application

**Terminal 1 - Start FastAPI server:**

```bash
cd server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 - Start Streamlit client:**

```bash
cd client
streamlit run app.py
```

The client will open at `http://localhost:8501`

## Usage

1. Open the Streamlit app in your browser
2. Type your medical question in the chat input
3. The system will:
   - Search relevant documents using vector similarity
   - Use chat history for context
   - Generate an answer with LLM
4. Continue the conversation—previous context is maintained

## API Endpoints

### POST `/ask/`

Ask a medical question with optional chat history.

**Request:**
```json
{
  "question": "What are the symptoms of diabetes?",
  "session_id": "optional-session-id",
  "chat_history": []
}
```

**Response:**
```json
{
  "answer": "...",
  "session_id": "session-uuid",
  "chat_history": [...],
  "status": "success"
}
```

## How RAG Works

1. **Indexing**: PDFs are converted to text, split into chunks, embedded with Google's model, and stored in Pinecone
2. **Retrieval**: User question → embedded → vector search → top 5 relevant documents retrieved
3. **Generation**: Retrieved context + chat history + question → LLM → answer

## Configuration

### Server-side
- **Embedding Model**: `gemini-embedding-2`
- **LLM Model**: `llama-3.3-70b-versatile`
- **Pinecone Region**: `us-east-1`
- **Chunk Size**: 500 characters (with 50 character overlap)
- **Retrieved Documents**: Top 5 relevant chunks

### Client-side
- **API URL**: `http://127.0.0.1:8000` (modify in `client/config.py`)

## Troubleshooting

**No PDFs found error:**
- Ensure PDF files are in `docs/` directory
- Run `store_vectorstore.py` to index them

**API Connection error:**
- Verify FastAPI server is running on port 8000
- Check API_URL in `client/config.py`

**Pinecone errors:**
- Verify API key and region are correct
- Check that the index is created in Pinecone console

**Missing embeddings:**
- Verify Google API key is valid
- Ensure sufficient API quota

## Technologies Used

- **Backend**: FastAPI, LangChain, Groq, Pinecone
- **Frontend**: Streamlit
- **Embeddings**: Google Generative AI
- **LLM**: Llama 3.3 via Groq
- **Vector DB**: Pinecone
