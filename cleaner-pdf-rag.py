# Same RAG pipeline as pdf-rag.py, but every model comes from Hugging Face (run locally)
# 1. Ingest the PDF file
# 2. Extract the text from the PDF and split it into chunks
# 3. Send the chunks to the embedding model to get embeddings
# 4. Store the embeddings in a vector database (e.g., FAISS, Pinecone, etc.)
# 5. Perform a similarity search in the vector database to find the most relevant chunks based on the user's query
# 6. Retrieve the similar documents and send them to the LLM along with the user's query to generate a response

import os
import warnings

import torch
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

warnings.filterwarnings("ignore")

DOC_PATH = "./data/m1-00-initial-pages.pdf"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_rag_hf"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_pdf(doc_path):
    """Ingest the PDF file."""
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"PDF file not found at {doc_path}. Please specify a valid path.")
    data = UnstructuredPDFLoader(doc_path).load()
    print(f"Loaded {len(data)} pages from the PDF file.")
    return data


def split_documents(data):
    """Split the extracted text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)
    print("Splitting is done. Total number of chunks created:", len(chunks))
    return chunks


def create_vector_db(chunks):
    """Embed the chunks and store them in the vector database."""
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": DEVICE}),
        collection_name=COLLECTION_NAME,
    )
    print("Added to vector database")
    return vector_db


def load_llm():
    """Load the Hugging Face chat model."""
    pipe = pipeline(
        "text-generation",
        model=MODEL,
        device=DEVICE,
        max_new_tokens=512,
        return_full_text=False,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))


def create_retriever(vector_db, llm):
    """Create a retriever that searches with multiple rephrasings of the user's question."""
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are a helpful assistant. Your task is to generate five different versions of the
        given user query to retrieve relevant documents from a vector database. By Generating multiple
        perspective on the user question, your goal is to help the user overcome some of the limitations
        of the distance-based similiarity search. Provide these alternative questions seperated by new lines.
        Original question: {question}
        """,
    )
    retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm=llm, prompt=query_prompt)
    print("Retriever is ready to use.")
    return retriever


def create_chain(retriever, llm):
    """Build the RAG chain: retrieve context, fill the prompt, ask the LLM."""
    template = """ Answer the question only based on the follwing context. {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain is ready to use.")
    return chain


def main():
    data = load_pdf(DOC_PATH)
    chunks = split_documents(data)
    vector_db = create_vector_db(chunks)

    llm = load_llm()
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)

    res = chain.invoke("What is the document about?")
    print(res)


if __name__ == "__main__":
    main()
