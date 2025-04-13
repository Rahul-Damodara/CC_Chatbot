import os
import streamlit as st
from dotenv import load_dotenv
from rag_system import build_rag_graph
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🤖",
    layout="wide"
)

# Add CSS for styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
    }
    .chat-message .message {
        padding-left: 1rem;
        padding-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_app" not in st.session_state:
        st.session_state.rag_app = build_rag_graph()

def load_vector_store():
    """Load the vector store."""
    persist_dir = "data/chroma_db"
    if not os.path.exists(persist_dir):
        st.error("Vector store not found. Please run ingest.py first to create the knowledge base.")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def display_chat_message(role, content):
    """Display a chat message."""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)

def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

def main():
    # Initialize session state
    initialize_session_state()
    
    # Page header
    st.title("🤖 RAG Chat System")
    st.markdown("Ask questions about your documents using Retrieval-Augmented Generation")
    
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This application uses LangGraph to implement a Retrieval-Augmented Generation (RAG) system.
        
        It retrieves relevant information from your documents and uses Groq's LLM to generate answers.
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Prepare state for RAG
        state = {
            "question": prompt,
            "chat_history": st.session_state.chat_history
        }
        
        # Run the RAG graph
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_app.invoke(state)
                
                # Update chat history
                st.session_state.chat_history = result["chat_history"]
                
                # Display assistant response
                assistant_response = result["generation"]
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                display_chat_message("assistant", assistant_response)
                
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()