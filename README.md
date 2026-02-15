# 🤖 Simple Bot  

A conversational AI chatbot built using **LangChain** and powered by **QwenBot (via Hugging Face Inference API)**.

🌐 **Live App:**  
👉 https://simplebot1526.streamlit.app/

---

## 🚀 About Simple Bot

**Simple Bot** is an interactive AI chatbot that allows users to have real-time conversations using a powerful large language model.

It combines:

- 🧠 **QwenBot** for intelligent language generation  
- 🔗 **LangChain** for structured prompt handling and conversation flow  
- 🎨 **Streamlit** for a clean and interactive web UI  
- ☁️ **Streamlit Cloud** for deployment  

---

## ✨ Features

- 💬 Real-time conversational interface  
- 🧠 Powered by QwenBot LLM  
- 🔗 Conversation history support  
- 🎛️ Adjustable temperature & token settings  
- ⚡ Hugging Face Inference API integration  
- ☁️ Cloud deployment ready  

---

## 🏗️ Tech Stack

- Python  
- Streamlit  
- LangChain  
- Hugging Face Hub  
- QwenBot  
- Pydantic  

---

## 📂 Project Structure

```
Langchain_prompt/
│
├── chatbot.py          # Main Streamlit chatbot app
├── messages.py         # Message handling
├── prompt_generator.py # Prompt utilities
├── prompt_ui.py        # Prompt-based UI tool
├── template.json       # Prompt templates
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AkhilK004/ChatBot-qwen.git
cd ChatBot-qwen
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Hugging Face Token

Create a `.env` file:

```
HF_TOKEN=your_huggingface_token_here
```

⚠️ Never commit your `.env` file.

### 5️⃣ Run the App

```bash
streamlit run chatbot.py
```

---

## 🔐 Environment Variable

| Variable | Description |
|----------|------------|
| HF_TOKEN | Hugging Face access token |

---

## 🧠 How It Works

1. User inputs a message in the Streamlit UI.
2. Messages are stored in session state.
3. LangChain structures the conversation.
4. The request is sent to QwenBot via Hugging Face.
5. The model generates a response.
6. The response is displayed in the chat UI.

---

## 🌍 Live Deployment

Deployed using **Streamlit Cloud**:

👉 https://simplebot1526.streamlit.app/

---

## 📌 Future Improvements

- Streaming token responses
- Model switching option
- Persistent database memory
- Conversation export feature
- User authentication

---

## 👨‍💻 Author

**Akhil K**  
GitHub: https://github.com/AkhilK004  

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
