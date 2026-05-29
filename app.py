import os

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


app = Flask(__name__)

load_dotenv()
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


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
