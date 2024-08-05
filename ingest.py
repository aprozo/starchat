import fitz  # PyMuPDF
import os
import arxiv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

def extract_text_and_metadata_from_pdfs(folder_path):
    data = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, filename))
            arxiv_id = filename.split("/")[-1].split(".pdf")[0]
            search = arxiv.Search(id_list=[arxiv_id])
            paper  = next(arxiv.Client().results(search))
            
            if paper:
                arxiv_metadata = {
                    "title": paper.title,
                    "categories" : '\n'.join([f'{i+1}. {cat}' for i, cat in enumerate(paper.categories)]),
                    "primary_category": paper.primary_category,
                    "published": paper.published.strftime('%Y-%m-%d'),
                    "authors": '\n'.join([f'{i+1}. {auth.name}' for i, auth in enumerate(paper.authors)])
                }
            else:
                arxiv_metadata = {
                    "title": doc.metadata.get("title", "Unknown"),
                    "categories": "Unknown",
                    "primary_category": "Unknown",
                    "published": doc.metadata.get("creationDate", "Unknown"),
                    "authors": doc.metadata.get("author", "Unknown")
                }

            print (arxiv_metadata)


            full_text = ""
            for page in doc:
                full_text += page.get_text()
                
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            chunks = text_splitter.split_text(full_text)
            
            for i, chunk in enumerate(chunks):
                data.append({
                    "id": f"{arxiv_id}_{i}",
                    "text": chunk,
                    "metadata": {
                        **arxiv_metadata,
                        "filename": filename,
                        "chunk_id": i
                    }
                })
    return data

data = extract_text_and_metadata_from_pdfs("test_docs")
texts = [item['text'] for item in data]
metadata_list = [item['metadata'] for item in data]
ids = [item['id'] for item in data]


model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(texts, show_progress_bar=True)

import chromadb
client = chromadb.Client()

collection = client.get_or_create_collection(name="pdfs")

collection.add(
    ids=ids,
    documents =texts,
    embeddings=embeddings,
    metadatas =metadata_list
)

results = collection.query(
    query_texts=["What are artcile categories?"],
    n_results=2
)

results