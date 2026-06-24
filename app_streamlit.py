"""
app_streamlit.py
-----------------
Streamlit web interface for the fine-tuned chatbot.

Run:
    streamlit run app_streamlit.py
"""

import streamlit as st
from src.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(page_title="Mental Health Support Chatbot", page_icon="💙")

st.title("💙 Mental Health Support Chatbot")
st.caption(
    "A fine-tuned DistilGPT2 demo trained on EmpatheticDialogues. "
    "This is a student project, not a medical device or crisis service."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot was fine-tuned on the **EmpatheticDialogues** dataset "
        "to practice giving gentle, supportive responses for everyday stress, "
        "anxiety, and emotional check-ins."
    )
    st.warning(
        "⚠️ Not a substitute for professional help. "
        "If you are in crisis, please contact a local emergency number "
        "or a crisis line such as 988 (US) or https://findahelpline.com."
    )


@st.cache_resource
def load_pipeline():
    return PredictionPipeline()


pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("How are you feeling today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking gently..."):
            reply = pipeline.generate_reply(user_input)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
