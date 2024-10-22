import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Streamlit page configuration
st.set_page_config(page_title="Document ChatBot", layout="wide")
st.title("📚 Document ChatBot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def initialize_env():
    """Initialize OpenAI API key from Streamlit secrets"""
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY

def load_documents(directory):
    """Load PDF documents from directory"""
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """Create vector store from document chunks"""
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_store"
    )
    return vector_store

def initialize_conversation_chain(vector_store):
    """Initialize conversation chain with memory"""
    llm = OpenAI(temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True
    )
    return conversation_chain

def main():
    initialize_env()
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Document Upload")
        pdf_dir = st.text_input("Enter PDF Directory Path:", "docs/")
        
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Load and process documents
                documents = load_documents(pdf_dir)
                chunks = split_documents(documents)
                vector_store = create_vector_store(chunks)
                st.session_state.conversation = initialize_conversation_chain(vector_store)
                st.success("Documents processed successfully!")

    # Main chat interface
    st.header("Chat Interface")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write("User: " + message["content"])
        else:
            st.write("Assistant: " + message["content"])

    # Chat input
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question and 'conversation' in st.session_state:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({"question": user_question})
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
            
            # Force streamlit to rerun to display new messages
            st.experimental_rerun()

    elif user_question and 'conversation' not in st.session_state:
        st.warning("Please process documents first!")

if __name__ == "__main__":
    main()