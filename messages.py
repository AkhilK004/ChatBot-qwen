from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import os
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation",
    huggingfacehub_api_token=os.environ["HF_TOKEN"]
)

model = ChatHuggingFace(
    llm=llm,
    temperature=1,
    max_new_tokens=60
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about the LangChain?")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content.strip()))

print(messages)