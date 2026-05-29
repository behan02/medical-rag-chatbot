import requests
from config import API_URL

def ask_question(question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    
    response = requests.post(f"{API_URL}/ask/", json=payload)
    return response