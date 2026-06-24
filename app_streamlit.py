"""
app_streamlit.py
-----------------
Streamlit web interface for the fine-tuned chatbot.

Run:
    streamlit run app_streamlit.py
"""

import streamlit as st
from src.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="💙",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Overall background */
        .stApp {
            background: linear-gradient(180deg, #f4f8fb 0%, #eef3f8 100%);
        }

        /* Header block */
        .hero {
            text-align: center;
            padding: 1.6rem 1rem 1.2rem 1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #a8d8ea 0%, #c9b6e4 100%);
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.07);
        }
        .hero h1 {
            font-size: 1.9rem;
            margin: 0;
            color: #2c3e50;
            font-weight: 700;
        }
        .hero p {
            margin-top: 0.4rem;
            color: #34495e;
            font-size: 0.95rem;
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 0.4rem 0.2rem;
            margin-bottom: 0.4rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #f0f4f9;
        }

        /* Chat input box */
        [data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
        }

        /* Spinner text */
        .stSpinner > div {
            text-align: center;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>💙 Mental Health Support Chatbot</h1>
        <p>A gentle space to talk through your day, your worries, or just how you feel right now.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 About this space")
    st.write(
        "This chatbot was fine-tuned on the **EmpatheticDialogues** dataset "
        "to practice giving gentle, supportive responses for everyday stress, "
        "anxiety, and emotional check-ins."
    )

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.write(
        "- Be as open as you'd like — this is a judgment-free space.\n"
        "- Responses are AI-generated and may not always make sense.\n"
        "- This is a student project, built for learning purposes."
    )

    st.markdown("---")
    st.warning(
        "⚠️ **Not a substitute for professional help.**\n\n"
        "If you are in crisis, please contact a local emergency number "
        "or a crisis line such as **988 (US)** or "
        "[findahelpline.com](https://findahelpline.com)."
    )

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()


pipeline = load_pipeline()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; color:#7f8c9b; padding: 2rem 1rem;">
            👋 <b>Say hello to start the conversation.</b><br>
            <span style="font-size:0.9rem;">e.g. "I've had a really stressful week" or "I'm feeling anxious about tomorrow"</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "💙"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

user_input = st.chat_input("How are you feeling today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="💙"):
        with st.spinner("Thinking gently..."):
            reply = pipeline.generate_reply(user_input)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})