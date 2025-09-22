from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv() # Moved load_dotenv here

PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_files(data="/content/MedicalChatBot/data")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_splitter(filter_data)
embedding = download_embeddings()




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
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name,
)
