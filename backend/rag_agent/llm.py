import os
from dotenv import load_dotenv
from openai import OpenAI
from backend.rag_agent.responsable import is_allowed_query, post_process_answer

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# SYSTEM prompt: professional financial assistant
SYSTEM = (
    "You are a professional finance assistant. Use the provided context to answer questions. "
    "If the context includes predictions, financial metrics, or market analysis, "
    "offer responsible, practical suggestions based on the information. "
    "Always include inline citations (Document Name, Section/Page) when using context. "
    "If the answer is not in the context, politely say 'I don’t know', and avoid speculation. "
    "Keep answers clear, concise, and professional, like a financial advisor would. "
    "Be friendly and helpful, guiding the user responsibly."
)

def answer(user_question: str, context: str, chat_history: list = None) -> str:
    """
    Generate a responsible financial assistant answer based on context.
    - user_question: The user's query.
    - context: Combined text from ingested documents.
    - chat_history: Optional previous conversation messages for context.
    """
    # Responsible AI: check query
    allowed, msg = is_allowed_query(user_question)
    if not allowed:
        return msg

    # Build prompt including chat history if available
    messages = [{"role": "system", "content": SYSTEM}]
    
    if chat_history:
        messages.extend(chat_history)

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n\n"
        "Instructions:\n"
        "- Answer like a professional financial assistant.\n"
        "- Use the context for predictions, metrics, or analysis.\n"
        "- Cite sources inline (Document Name, Section/Page).\n"
        "- If info is missing, politely say 'I don't know'."
    )

    messages.append({"role": "user", "content": user_prompt})

    # Generate response from OpenAI
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2  # slightly creative, professional guidance
    )

    raw_answer = resp.choices[0].message.content.strip()
    return post_process_answer(user_question, raw_answer)
