import os
import arxiv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader

from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

# =============================================================================
# ===========================Read PDFs=========================================

def getArxivPaper(name):
     print (f"Searching for the paper with ID: {name}")
     if len(name) == 7:
          name = "nucl-ex/" + name
     try:
          search = arxiv.Search( id_list=[name] )
          paper = next(arxiv.Client().results(search))
          return paper
     
     except StopIteration:
          print(f"No results found for the ID: {name}")

     # if length of name is 7, then it is an old arxiv id , add prefix nucl-ex or hep-ex if nucl-ex was not found

     if name.startswith("nucl-ex/"):
          name = name.replace("nucl-ex/","hep-ex/")
          return getArxivPaper(name)
     
     return None

def getDocument(filename):
    base_filename = filename.split("/")[-1].split(".pdf")[0]
    # if arxiv_id does not have a dot, add nucl-ex/ in front of it
    paper = getArxivPaper(base_filename)
    text_metadata = {
        "title": paper.title,
        "published": paper.published.strftime('%Y-%m-%d'),
        "authors": '\n'.join([f'{i+1}. {auth.name}' for i, auth in enumerate(paper.authors)])
    }

    text_metadata["arxiv_id"] = paper.get_short_id()

    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    full_text = ""
    page_number = 1
    for page in pages:
        full_text += f"\n PAGE {page_number}\n"
        full_text += page.page_content
        page_number += 1
        
    document= Document(content=full_text, metadata=text_metadata) 
    print (f"Document {base_filename} has been loaded")
    return document





# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# =============================================================================
# ===========================Store Document====================================

import chromadb

persistent_client = chromadb.PersistentClient(path="database")
collection = persistent_client.get_or_create_collection("arxiv_papers")

child_vectorstore = Chroma(
    client=persistent_client,
    embedding_function=embedding_function
)

# # The storage layer for the parent documents
parent_docstore = InMemoryStore()

child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore, 
    docstore=parent_docstore,
    child_splitter=child_splitter)



folder_path="arxiv_papers"
for filename in tqdm(os.listdir(folder_path)):
    if not filename.endswith(".pdf"):
        continue
    filename = os.path.join(folder_path, filename)
    print (f"Loading {filename}")
    print 
    document = getDocument(filename)
    retriever.add_documents([document])
# =============================================================================
# ===========================Split Document====================================
