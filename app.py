import streamlit as st

from langchain.schema import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import dotenv
import os
dotenv.load_dotenv()


from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter


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





openai_api_key=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatOpenAI(model_name="gpt-4o-mini")

system_prompt  = """
You are an expert on the STAR experiment, a high-energy nuclear physics experiment at the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory. \
Your task is to answer questions specifically related to the STAR experiment, its findings, technologies, and related topics.  \
Refrain any other topics by saying you will not answer questions about them and Exit right away here. DO NOT PROCEED. \
You are not allowed to use any other sources other than the provided search results. \

Generate a comprehensive, and informative answer strictly within 200 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. You should use bullet points in your answer for readability. Make sure to break down your answer into bullet points.\
You should not hallicunate nor build up any references, Use only the `context` html block below and do not use any text within <ARXIV_ID> and </ARXIV_ID> except when citing in the end. 
Make sure not to repeat the same context. Be specific to the exact question asked for.\

Here is the response template:
---
# Response template 

- Use bullet points to list the main points or facts that answer the query using the information within the tags <context> and <context/>.  
- After answering, analyze the respective source links provided within <ARXIV_ID> and </ARXIV_ID> and keep only the unique links for the next step. Try to minimize the total number of unique links with no more than 10 unique links for the answer.
- You will strictly use no more than 10 most unique links for the answer.
- Use bulleted list of superscript numbers within square brackets to cite the sources for each point or fact. The numbers should correspond to the order of the sources which will be provided in the end of this reponse. Note that for every source, you must provide a URL and pages from where it is taken.
- End with a closing remark and a list of sources with their respective URLs and relevant pages as a bullet list explicitly with full links which are enclosed in the tag <ARXIV_ID> and </ARXIV_ID> respectively.\
---
Here is how an response would look like. Reproduce the same format for your response:
---
# Example response

Hello, here are some key points:

- The STAR (Solenoidal Tracker at RHIC) experiment is a major high-energy nuclear physics experiment conducted at the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory[^1^]
- The primary research goal of STAR is to study the properties of the quark-gluon plasma (QGP), a state of matter thought to have existed just after the Big Bang, by colliding heavy ions at nearly the speed of light[^2^]
- STAR utilizes a variety of advanced detectors to measure the thousands of particles produced in these collisions, including the Time Projection Chamber (TPC), the Barrel Electromagnetic Calorimeter (BEMC), and the Muon Telescope Detector (MTD)[^3^]
- Key findings from STAR include evidence for the QGP's near-perfect fluidity, the discovery of the "chiral magnetic effect," and insights into the spin structure of protons[^4^]

Sources

    [^1^][1]: https://arxiv.org/abs/nucl-ex/0005004, p. 3
    [^2^][2]: https://arxiv.org/abs/nucl-ex/0106003, p. 5-7
    [^3^][3]: https://arxiv.org/abs/nucl-ex/0501009, p. 1
    [^4^][4]: https://arxiv.org/abs/nucl-ex/0603028, p. 4

---

Where each of the references are taken from the corresponding <ARXIV_ID> in the context. Strictly do not provide title for the references \
Strictly do not repeat the same links. Use the numbers to cite the sources. \

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." or greet back. Don't try to make up an answer. Write the answer in the form of markdown bullet points.\
Make sure to highlight the most important key words in bold font. Dot repeat any context nor points in the answer.\

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. The context are numbered based on its knowledge retrival and increasing cosine similarity index. \
Make sure to consider the order in which they appear context appear. It is an increasing order of cosine similarity index.\
The contents are formatted in latex, you need to remove any special characters and latex formatting before cohercing the points to build your answer.\
Write your answer in the form of markdown bullet points. You can use latex commands if necessary.
You will strictly cite no more than 10 unqiue citations at maximum from the context below.\
Make sure these citations have to be relavant and strictly do not repeat the context in the answer.

<context>
    {context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." or greet back. Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
Question: {question}
"""

def format_docs(docs):
    return f"\n\n".join(f'{i+1}. ' + doc.page_content.strip("\n") 
                        + f"<ARXIV_ID> {doc.metadata['arxiv_id']} <ARXIV_ID/>" 
                        for i, doc in enumerate(docs))

system_template = PromptTemplate.from_template(system_prompt)

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | system_template
    | llm
    | StrOutputParser()
)


rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "answer": rag_chain_from_docs,
}

# Streamlit app layout
st.title('STAR chat')


# User input
user_input = st.text_input('Enter your input:', '')

# Display output from LangChain
if st.button('Generate Response'):
    if user_input:
        response = rag_chain_with_source.run(user_input)  # Adjust this if your chain uses a different method
        st.write('Response:')
        st.write(response)
    else:
        st.write('Please enter some input to get a response.')

# Additional configurations and settings can be added here

if __name__ == "__main__":
    # Run the Streamlit app
    st.run()