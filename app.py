from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from flipkart.retrieval_generation import generation, get_session_history
from flipkart.data_ingestion import data_ingestion
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize model
model = ChatGroq(model="llama3-70b-8192", temperature=0.5)

# Chat history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]  # ❗ Fix: use square brackets, not function call

# Load vector store (once only)
vstore = data_ingestion("done")

# Create RAG chain
chain = generation(vstore)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST", "GET"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]

        result = chain.invoke(
            {"input": msg},
            config={
                "configurable": {"session_id": "dhruv"}  # static session for now
            },
        )["answer"]

        return str(result)
    return "GET not supported here."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
