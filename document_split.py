import os

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"), 
                                    port=int(os.getenv("CHROMA_PORT")), 
                                    settings=Settings())

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

file_path = "arxiv_papers"

for file in os.listdir(file_path):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(document)

    Chroma.from_documents(
        documents=chunked_documents,
        embedding=embedding_function,
        collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
        client=chroma_client,
    )
    print(f"Added {len(chunked_documents)} chunks to chroma db")
