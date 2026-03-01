import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Embedded CSS (modern, clean, no external frameworks) ----------------
st.markdown(
    """
<style>
/* Hide Streamlit chrome */
header, footer { visibility: hidden; }

/* Center content and add breathing room */
.block-container {
  max-width: 980px;
  padding-top: 1.75rem;
  padding-bottom: 2rem;
}

/* Hero */
.hero-wrap { display:flex; flex-direction:column; align-items:center; gap:8px; margin-bottom: 14px; }
.hero {
  font-size: 44px;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.05;
  text-align: center;
}
.subhero {
  opacity: 0.7;
  font-size: 15px;
  text-align: center;
  margin-bottom: 6px;
}

/* Chips row */
.chips-row { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin: 10px 0 18px; }

/* Make buttons look like chips */
div.stButton > button {
  border-radius: 999px !important;
  padding: 0.55rem 0.9rem !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}

/* Chat input rounding */
div[data-testid="stChatInput"] textarea {
  border-radius: 14px !important;
}

/* Subtle card around chat area */
.chat-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 16px;
  background: rgba(255,255,255,0.03);
}

/* Empty state */
.empty-state {
  border: 1px dashed rgba(255,255,255,0.18);
  border-radius: 16px;
  padding: 18px 16px;
  opacity: 0.85;
}
.empty-title { font-weight: 700; font-size: 16px; margin-bottom: 6px; }
.empty-list { margin: 0; padding-left: 18px; opacity: 0.9; }
.small-muted { opacity: 0.65; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Token ----------------
token = os.environ.get("HF_TOKEN")
if not token:
    st.error("HF_TOKEN not found. Put it in your .env as HF_TOKEN=...")
    st.stop()

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Settings")

    repo_id = st.text_input("Model repo_id", value="Qwen/Qwen3-Coder-Next")
    provider = st.selectbox("Provider", ["novita", "auto"], index=0)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_new_tokens = st.slider("Max new tokens", 10, 2000, 300, 50)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]
            st.rerun()

    with col_b:
        # Optional: Export chat
        def _export_payload():
            msgs = st.session_state.get("chat_history", [SystemMessage(content="You are a helpful assistant.")])
            out = []
            for m in msgs:
                role = "system"
                if isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                out.append({"role": role, "content": m.content})
            return json.dumps({"messages": out}, indent=2)

        st.download_button(
            "⬇️ Export",
            data=_export_payload(),
            file_name="chat.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption("Tip: lower temperature for factual answers, higher for creative writing.")

# ---------------- Model factory (cached per settings) ----------------
@st.cache_resource
def get_model(repo_id: str, provider: str, temperature: float, max_new_tokens: int):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        huggingfacehub_api_token=token,
        provider=provider,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return ChatHuggingFace(llm=llm)

model = get_model(repo_id, provider, temperature, max_new_tokens)

# ---------------- Session state chat history ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

# ---------------- Hero ----------------
st.markdown(
    """
<div class="hero-wrap">
  <div class="hero">What can I help with?</div>
  <div class="subhero">Ask anything — powered by Qwen via Hugging Face endpoint + LangChain.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- Starter prompt chips (send a message) ----------------
chip_cols = st.columns(4)
chips = [
    "Explain LangChain in simple words with an example.",
    "Write a short professional email for a leave request.",
    "Help me debug this Python error: ...",
    "Generate a study plan for my exam in 7 days.",
]
for i, text in enumerate(chips):
    with chip_cols[i]:
        if st.button(text, use_container_width=True):
            st.session_state.pending_prompt = text

# ---------------- Chat input placed under hero ----------------
pending = st.session_state.pop("pending_prompt", None)
user_input = st.chat_input("Message…")

# If chip pressed, treat as user input
if pending and not user_input:
    user_input = pending

# ---------------- Helper: streaming invoke with fallback ----------------
def generate_answer(history):
    # Try streaming if supported; fallback to invoke
    try:
        chunks = []
        for chunk in model.stream(history):
            piece = getattr(chunk, "content", "") or ""
            chunks.append(piece)
            yield "".join(chunks)
        return
    except Exception:
        resp = model.invoke(history)
        yield resp.content.strip()

# ---------------- Handle user message ----------------
if user_input:
    txt = user_input.strip()
    if txt.lower() in {"exit", "quit", "end"}:
        with st.chat_message("assistant"):
            st.markdown("Goodbye! 👋")
        st.stop()

    st.session_state.chat_history.append(HumanMessage(content=txt))

    # Render user's message immediately
    with st.chat_message("user"):
        st.markdown(txt)

    # Assistant response (stream + fallback)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        final_text = ""
        with st.spinner("Thinking..."):
            try:
                for partial in generate_answer(st.session_state.chat_history):
                    final_text = partial
                    placeholder.markdown(final_text)
            except Exception as e:
                final_text = f"Error: {e}"
                placeholder.markdown(final_text)

    st.session_state.chat_history.append(AIMessage(content=final_text.strip()))
    st.rerun()

# ---------------- Render messages (below input) ----------------
# Wrap chat area in subtle card
st.markdown('<div class="chat-card">', unsafe_allow_html=True)

rendered_any = False
for msg in st.session_state.chat_history:
    if isinstance(msg, SystemMessage):
        continue
    rendered_any = True
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if not rendered_any:
    st.markdown(
        """
<div class="empty-state">
  <div class="empty-title">Try asking:</div>
  <ul class="empty-list">
    <li>“Explain transformers like I’m new to ML.”</li>
    <li>“Summarize this text: …”</li>
    <li>“Write code to parse a CSV and plot results.”</li>
    <li>“Help me prepare for an interview in 3 days.”</li>
  </ul>
  <div class="small-muted" style="margin-top:10px;">Use the chips above for quick starts.</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)