import os
import glob
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_documents(directory: str) -> List[Document]:
    """Load documents from a directory."""
    documents = []
    
    # Load text files
    for file_path in glob.glob(os.path.join(directory, "*.txt")):
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    
    # Load PDF files
    for file_path in glob.glob(os.path.join(directory, "*.pdf")):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document], persist_directory: str) -> None:
    """Create and persist a vector store from documents."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    print(f"Vector store created with {len(documents)} chunks and persisted to {persist_directory}")

def main():
    # Define directories
    data_dir = "data/documents"
    persist_dir = "data/chroma_db"
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)
    
    # Load documents
    print("Loading documents...")
    documents = load_documents(data_dir)
    
    if not documents:
        print(f"No documents found in {data_dir}. Please add some .txt or .pdf files.")
        return
    
    # Split documents
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    # Create vector store
    print("Creating vector store...")
    create_vector_store(chunks, persist_dir)

if __name__ == "__main__":
    main()