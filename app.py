from openai import OpenAI
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

import os
import time
import dotenv
dotenv.load_dotenv()

embeddings = OpenAIEmbeddings()


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return f"\n\n".join(f'{i+1}. ' + doc.page_content.strip("\n") + f"<ARXIV_ID> {doc.metadata['arxiv_id']} <ARXIV_ID/>" for i, doc in enumerate(docs))


from langchain.prompts import PromptTemplate



template = """
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

- Start with a greeting and a summary of the user's query
- Use bullet points to list the main points or facts that answer the query using the information within the tags <context> and <context/>.  
- After answering, analyze the respective source links provided within <ARXIV_ID> and </ARXIV_ID> and keep only the unique links for the next step. Try to minimize the total number of unique links with no more than 10 unique links for the answer.
- You will strictly use no more than 10 most unique links for the answer.
- Use bulleted list of superscript numbers within square brackets to cite the sources for each point or fact. The numbers should correspond to the order of the sources which will be provided in the end of this reponse. Note that for every source, you must provide a URL.
- End with a closing remark and a list of sources with their respective URLs as a bullet list explicitly with full links which are enclosed in the tag <ARXIV_ID> and </ARXIV_ID> respectively.\
---
Here is how an response would look like. Reproduce the same format for your response:
---
# Example response

Hello, thank you for your question about the STAR experiment. Here are some key points about STAR:

- The STAR (Solenoidal Tracker at RHIC) experiment is a major high-energy nuclear physics experiment conducted at the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory[^1^]
- The primary research goal of STAR is to study the properties of the quark-gluon plasma (QGP), a state of matter thought to have existed just after the Big Bang, by colliding heavy ions at nearly the speed of light[^2^]
- STAR utilizes a variety of advanced detectors to measure the thousands of particles produced in these collisions, including the Time Projection Chamber (TPC), the Barrel Electromagnetic Calorimeter (BEMC), and the Muon Telescope Detector (MTD)[^3^]
- Key findings from STAR include evidence for the QGP's near-perfect fluidity, the discovery of the "chiral magnetic effect," and insights into the spin structure of protons[^4^]

I hope this helps you understand more about the STAR experiment.
Sources

    [^1^][1]: https://arxiv.org/abs/nucl-ex/0005004
    [^2^][2]: https://arxiv.org/abs/nucl-ex/0106003
    [^3^][3]: https://arxiv.org/abs/nucl-ex/0501009
    [^4^][4]: https://arxiv.org/abs/nucl-ex/0603028

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


rag_prompt_custom = PromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "answer": rag_chain_from_docs,
}

st.title("STAR chat")

st.sidebar.title("Data Collection")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        allchunks = None
        with st.spinner("Gathering info from Knowledge Bank and writing response..."):
            allchunks = rag_chain_with_source.stream(prompt)
            message_placeholder = st.empty()
            for chunk in allchunks:
                full_response += (chunk.get("answer") or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})