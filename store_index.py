import os
from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_files("/content/MedicalChatBot/data")
filter_data = filter_to_minimal_docs(extracted_data)
texts_chunk = text_splitter(filter_data)
embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
  pc.create_index(
      name=index_name,
      dimension=384,#dimension of the embeddings
      metric="cosine",#cosine similarity
      spec=ServerlessSpec(cloud="aws", region="us-east-1"),

  )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    index_name=index_name,
    embedding=embedding,
)
