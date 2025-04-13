import os
from dotenv import load_dotenv
from rag_system import build_rag_graph
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

def main():
    # Check if vector store exists
    persist_dir = "data/chroma_db"
    if not os.path.exists(persist_dir):
        print("Vector store not found. Please run ingest.py first to create the knowledge base.")
        return
    
    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # Build the RAG graph
    rag_app = build_rag_graph()
    
    # Initialize chat history
    chat_history = []
    
    print("RAG Chat System (type 'exit' to quit)")
    print("-------------------------------------")
    
    while True:
        # Get user input
        question = input("\nYou: ")
        
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        # Prepare state
        state = {
            "question": question,
            "chat_history": chat_history
        }
        
        # Run the graph
        try:
            result = rag_app.invoke(state)
            
            # Update chat history
            chat_history = result["chat_history"]
            
            # Print the answer
            print(f"\nAI: {result['generation']}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()