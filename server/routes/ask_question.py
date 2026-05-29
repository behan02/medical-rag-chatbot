from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from logger import logger
from modules.load_vectorstore import get_retriever
from modules.llm import get_llm_chain
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from datetime import datetime

router = APIRouter()

# In-memory chat history storage (key: session_id, value: list of messages)
chat_store: Dict[str, List[dict]] = {}


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    chat_history: Optional[List[dict]] = None


@router.post("/ask/")
async def ask_question(request: QuestionRequest):
    """
    Answer a medical question using RAG with persistent server-side chat history.
    
    Args:
        request: QuestionRequest with 'question', optional 'session_id', and optional 'chat_history'
    
    Returns:
        JSON with answer, updated chat history, and session_id
    
    Flow:
    1. Creates or retrieves session-based chat history
    2. Uses existing history for context (if available)
    3. Generates answer using RAG + LLM chain
    4. Automatically stores new Q&A in server-side history
    5. Returns answer + updated history to client
    """
    try:
        # Generate or use provided session_id
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"Session ID: {session_id}")
        logger.info(f"User query: {request.question}")

        # Initialize or retrieve chat history from server storage
        if session_id not in chat_store:
            chat_store[session_id] = []
        
        server_chat_history = chat_store[session_id]
        
        # Convert server-stored history to LangChain message format
        langchain_history = []
        for msg in server_chat_history:
            if msg.get("role") == "user":
                langchain_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                langchain_history.append(AIMessage(content=msg.get("content", "")))

        logger.info(f"📜 Using {len(server_chat_history)} previous messages for context")

        retriever = get_retriever()
        chain = get_llm_chain(retriever)

        result = chain.invoke({
            "input": request.question,
            "chat_history": langchain_history
        })

        server_chat_history.append({
            "role": "user",
            "content": request.question,
            "timestamp": datetime.now().isoformat()
        })
        
        server_chat_history.append({
            "role": "assistant",
            "content": result,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"✅ Stored Q&A in history. Total messages: {len(server_chat_history)}")

        return JSONResponse(
            status_code=200,
            content={
                "answer": result,
                "session_id": session_id,
                "chat_history": server_chat_history,
                "status": "success"
            }
        )

    except Exception as e:
        logger.exception(f"❌ Error processing question: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "error"}
        )


@router.delete("/chat/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a specific session.
    
    Args:
        session_id: The session ID to clear
    
    Returns:
        Confirmation message
    """
    if session_id in chat_store:
        chat_store[session_id] = []
        logger.info(f"✅ Cleared chat history for session: {session_id}")
        return JSONResponse(
            status_code=200,
            content={"message": f"Chat history cleared for session {session_id}", "status": "success"}
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"Session {session_id} not found", "status": "error"}
        )