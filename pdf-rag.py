# 1. Ingest the PDF file
# 2. Extract the text from the PDF and split it into chunks
# 3. Send the chunks to the embedding model to get embeddings
# 4. Store the embeddings in a vector database (e.g., FAISS, Pinecone, etc.)
# 5. Perform a similarity search in the vector database to find the most relevant chunks based on the user's query
# 6. Retrieve the similar documents and send them to the LLM along with the user's query to generate a response


import warnings
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
warnings.filterwarnings("ignore")

doc_path = "./data/m1-00-initial-pages.pdf"
model = "qwen2.5:latest"

if doc_path:
    loader = UnstructuredPDFLoader(doc_path)
    data = loader.load()
    print(f"Loaded {len(data)} pages from the PDF file.")
else:
    print("No PDF file path provided. Please specify a valid path to the PDF file.")

# End of PDf file ingestion.

# Extract the text from the PDF and split it into chunks
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("Splitting is done. Total number of chunks created:", len(chunks))

# End of chunking the text.

# add to vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="pdf_rag",
)
print("Added to vector database")
# Retrieval of similar documents based on user query
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever

llm = ChatOllama(model=model)
QueryPrompt = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a helpful assistant. Your task is to generate five different versions of the 
    given user query to retrieve relevant documents from a vector database. By Generating multiple 
    perspective on the user question, your goal is to help the user overcome some of the limitations
    of the distance-based similiarity search. Provide these alternative questions seperated by new lines.
    Original question: {query}
    """,
)

retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm=llm, prompt=QueryPrompt)

print("Retriever is ready to use.")

# RAG prompt
template = """ Answer the question only based on the follwing context. {context}
Question: {query}"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("RAG chain is ready to use.")

res = chain.invoke(input=("What is the document about?",))

print(res)