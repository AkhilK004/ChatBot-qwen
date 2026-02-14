import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

st.set_page_config(page_title="LangChain Chat UI", page_icon="💬")
st.title("💬 Chatbot (HuggingFace + LangChain)")

token = os.environ.get("HF_TOKEN")
if not token:
    st.error("HF_TOKEN not found. Put it in your .env as HF_TOKEN=...")
    st.stop()

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Settings")
    repo_id = st.text_input("Model repo_id", value="Qwen/Qwen3-Coder-Next")
    provider = st.selectbox("Provider", ["novita", "auto"], index=0)
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    max_new_tokens = st.slider("Max new tokens", 10, 500, 60, 10)

    if st.button("🧹 Clear chat"):
        st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]
        st.rerun()

# ---------------- Model factory (cached per settings) ----------------
@st.cache_resource
def get_model(repo_id: str, provider: str, temperature: float, max_new_tokens: int):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        huggingfacehub_api_token=token,
        provider=provider,
        # generation params belong here (not on ChatHuggingFace)
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return ChatHuggingFace(llm=llm)

model = get_model(repo_id, provider, temperature, max_new_tokens)

# ---------------- Session state chat history ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Tell me about LangChain."),
    ]
    # Generate initial assistant response once
    try:
        first = model.invoke(st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=first.content.strip()))
    except Exception as e:
        st.session_state.chat_history.append(AIMessage(content=f"Error: {e}"))

# ---------------- Render messages ----------------
for msg in st.session_state.chat_history:
    if isinstance(msg, SystemMessage):

        with st.chat_message("system"):
            st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---------------- Input + invoke ----------------
user_input = st.chat_input("Type your message... (exit/quit/end to stop)")

if user_input:
    if user_input.lower().strip() in ["exit", "quit", "end"]:
        with st.chat_message("assistant"):
            st.markdown("Goodbye! 👋")
        st.stop()

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = model.invoke(st.session_state.chat_history)
                answer = resp.content.strip()
            except Exception as e:
                answer = f"Error: {e}"

            st.markdown(answer)

    st.session_state.chat_history.append(AIMessage(content=answer))
