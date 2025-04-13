import os
from typing import Dict, List, Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define state
class GraphState(TypedDict):
    question: str
    context: List[Document]
    generation: str
    chat_history: List[Dict]

# Initialize vector store
def init_vector_store(docs_path="data/documents"):
    # This is a placeholder. In a real application, you would load documents
    # from files or a database and split them into chunks.
    documents = [
        Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.", 
                 metadata={"source": "langchain docs"}),
        Document(page_content="RAG stands for Retrieval Augmented Generation, a technique to enhance LLM outputs with external data.", 
                 metadata={"source": "AI research paper"}),
        Document(page_content="Python is a programming language known for its readability and versatility.", 
                 metadata={"source": "programming guide"})
    ]
    
    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents, embeddings)
    
    return vector_store

# Initialize components
vector_store = init_vector_store()
llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)

# Define the retriever node
def retrieve(state: GraphState) -> GraphState:
    """Retrieve relevant documents based on the question."""
    question = state["question"]
    docs = vector_store.similarity_search(question, k=5)
    return {"context": docs}

# Define the generation node
def generate(state: GraphState) -> GraphState:
    """Generate an answer based on the context and question."""
    context = state["context"]
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer the user's question. "
                  "If you don't know the answer, just say so. Don't make up information.\n\n"
                  "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Format context as string
    context_str = "\n\n".join([doc.page_content for doc in context])
    
    # Generate response
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({
        "context": context_str,
        "chat_history": chat_history,
        "question": question
    })
    
    return {"generation": generation}

# Define the update history node
def update_history(state: GraphState) -> GraphState:
    """Update chat history with the new question and answer."""
    chat_history = state.get("chat_history", [])
    chat_history.append({"role": "user", "content": state["question"]})
    chat_history.append({"role": "assistant", "content": state["generation"]})
    return {"chat_history": chat_history}

# Build the graph
def build_rag_graph():
    # Initialize the graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("update_history", update_history)
    
    # Add edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "update_history")
    graph.add_edge("update_history", END)
    
    # Set entry point
    graph.set_entry_point("retrieve")
    
    # Compile the graph
    return graph.compile()

# Create the RAG application
rag_app = build_rag_graph()

# Example usage
if __name__ == "__main__":
    # Initial state
    initial_state = {"question": "What is LangGraph?", "chat_history": []}
    
    # Run the graph
    result = rag_app.invoke(initial_state)
    
    # Print the result
    print("Question:", initial_state["question"])
    print("Answer:", result["generation"])
    
    # Ask a follow-up question
    follow_up_state = {
        "question": "How is it related to RAG?",
        "chat_history": result["chat_history"]
    }
    
    # Run the graph again
    follow_up_result = rag_app.invoke(follow_up_state)
    
    # Print the follow-up result
    print("\nFollow-up Question:", follow_up_state["question"])
    print("Answer:", follow_up_result["generation"])