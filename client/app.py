import streamlit as st
from components.chatUI import render_chat

st.set_page_config(page_title="AI Medical Assistant", layout="centered")
st.title("🩺 Medical Assistant Chatbot")

render_chat()