import os
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from src.config import load_settings
from src.helper import download_embeddings
from src.prompt import system_prompt
from src.config import load_settings
from src.helper import download_embeddings
from src.prompt import system_prompt
app = Flask(__name__)
load_dotenv()
app = Flask(__name__)
settings = load_settings()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_embeddings()

# Load vectors created by scripts/ingest_books.py from Qdrant.
docsearch = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=settings.qdrant_collection,
    url=settings.qdrant_url,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    model="writer/palmyra-med-70b-32k",
    temperature=0.2,
    top_p=0.7,
    max_tokens=32000,
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
@lru_cache(maxsize=1)
def get_rag_chain():
    """Build the retrieval chain lazily so Render health checks stay lightweight."""

    settings = load_settings()
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    embeddings = download_embeddings()
    docsearch = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": settings.retrieval_k}
    )

    chat_model = ChatOpenAI(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0.2,
        top_p=0.7,
        max_tokens=4096,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "service": "medical-chatbot"})
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a medical question so I can help.", 400

    try:
        response = get_rag_chain().invoke({"input": msg})
    except Exception:  # pragma: no cover - defensive runtime guard for hosted envs
        app.logger.exception("RAG chain failed")
        return (
            "I’m having trouble connecting to the medical knowledge base right now. "
            "Please verify the Qdrant URL, collection, and API keys in your Render environment."
        ), 503

    return str(response.get("answer", "I could not generate an answer for that question."))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
