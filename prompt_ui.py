import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace



load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox(
    "Select a research paper",
    ["Attention is all you need", "BERT:Pre-training of Deep learning", "GPT-3:Language Models are Few-Shot Learners","Pokemon","Anime :Attack on Titan"]
)

style_input = st.selectbox(
    "Select a style",
    ["Formal", "Informal", "Technical", "Simple", "friendly"]
)

length_input = st.slider(
    "Select the length of the summary",
    min_value=50, max_value=500, value=200, step=10
)

template = load_prompt('template.json')

prompt = template.invoke(
    {
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    }
)


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],  # make sure HF_TOKEN is set in .env
    provider="novita",  # or "auto"
)

model = ChatHuggingFace(
    llm=llm,
    temperature=1,
    max_new_tokens=200
)

if st.button("Submit"):
    result = model.invoke(prompt)   # pass a plain string
    st.write(result.content.strip())
