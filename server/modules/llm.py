from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever):
    llm = get_llm()

    # --- Step 1: Contextualize question using chat history ---
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Given a chat history and the latest user question which might reference "
         "context in the chat history, formulate a standalone question which can be "
         "understood without the chat history. Do NOT answer the question, just "
         "reformulate it if needed and otherwise return it as is."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    # --- Step 2: QA prompt ---
    qa_system_prompt = """You are **MediBot**, an AI-powered medical assistant trained to help users \
understand medical documents and health-related questions.

Your responsibilities:
- Provide clear, accurate, and helpful responses based ONLY on the provided context
- Respond in a calm, factual, and respectful tone
- Use simple explanations when needed
- If the context does not contain the answer, respond: "I'm sorry, but I couldn't find relevant information in the provided documents."
- Do NOT make up facts or provide medical diagnoses
- Do NOT give personal medical advice
- Cite relevant sections from the documents when appropriate

Context from documents:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # --- Step 3: Compose the full chain cleanly ---
    def get_contextualized_question(x: dict) -> str:
        if x.get("chat_history"):
            return contextualize_chain.invoke(x)
        return x["input"]

    rag_chain = (
        RunnablePassthrough.assign(
            # Reformulate question if needed, store as standalone_question
            standalone_question=RunnableLambda(get_contextualized_question)
        )
        | RunnablePassthrough.assign(
            # Retrieve docs using the standalone question
            context=lambda x: _format_docs(retriever.invoke(x["standalone_question"]))
        )
        | qa_prompt  
        | llm
        | StrOutputParser()
    )

    return rag_chain


def get_llm_chain(retriever):
    return create_rag_chain(retriever)