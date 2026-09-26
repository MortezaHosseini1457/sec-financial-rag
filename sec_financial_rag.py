
# Imports
import os

from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from edgar import Company, set_identity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

DAHL_API_KEY = os.getenv("DAHL_API_KEY")


# SEC Identity

set_identity("morteza hosseini smhsmh1998@gmail.com")

# Load SEC Filing
filing = Company("NVDA").get_filings(form="10-K").latest()
md = filing.markdown()
elements = partition_md(
        text=md
    )

# Parse Document
chunks = chunk_by_title(
        elements,
        max_characters=1500,
        new_after_n_chars=1200,
        overlap=100,
    )

# Chunk Document
documents = []

for chunk in chunks:
    documents.append({
        "text": chunk.text,
        "metadata": chunk.metadata.to_dict()
    })


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Convert chunks to LangChain Documents
langchain_documents = [
    Document(
        page_content=doc["text"],
        metadata=doc["metadata"]
    )
    for doc in documents
]

vector_store = FAISS.from_documents(
    langchain_documents,
    embeddings
)

# Retrieval

def retrieve_context(query, k=5):
    docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join(
    f"""
    SOURCE:
    {doc.metadata}

    CONTENT:
    {doc.page_content}
    """
    for doc in docs
    )

    return context


llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",
    api_key=DAHL_API_KEY ,
    base_url="https://inference.dahl.global/v1",
)

# RAG Pipeline

def ask_rag(question):

    context = retrieve_context(question)

    prompt = f"""
    You are a financial research assistant.

    Answer the question using ONLY the provided context.

    For every factual claim, identify the source chunk that supports it.
    If the answer requires calculation, show the calculation briefly.

    If the information is not available in the context, say:
    "I don't have enough information in the provided documents."

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response.content

answer = ask_rag(
    "What are the main risks NVIDIA faces according to its 10-K?"
)

print(answer)
