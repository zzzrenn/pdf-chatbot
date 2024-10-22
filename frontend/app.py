# frontend/app.py
import streamlit as st
import requests
from typing import Dict, List
import os
import base64
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_document" not in st.session_state:
        st.session_state.selected_document = None
    if "documents" not in st.session_state:
        st.session_state.documents = []

class DocumentManager:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
    
    def upload_document(self, file) -> bool:
        """Upload a document to the backend"""
        try:
            files = {"file": (file.name, file, "application/pdf")}
            response = requests.post(f"{self.api_url}/upload", files=files)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Error uploading document: {str(e)}")
            return False
    
    def get_documents(self) -> List[Dict]:
        """Fetch list of documents from backend"""
        response = requests.get(f"{self.api_url}/documents")
        if response.status_code == 200:
            return response.json()
        st.error(f"Error fetching documents: ERROR {response.status_code}")
        return []
            
    
    def get_document_content(self, filename: str) -> bytes:
        """Fetch document content from backend"""
        response = requests.get(f"{self.api_url}/document/{filename}")
        if response.status_code == 200:
            return response.content
        st.error(f"Error fetching document content: ERROR {response.status_code}")
        return None

class ChatInterface:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
    
    def get_response(self, question: str) -> Dict:
        """Get response from chatbot"""
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json={"question": question}
            )
            return response.json()
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
            return {"answer": "Error: Could not get response", "source_documents": []}

def display_pdf(pdf_content: bytes):
    """Display PDF content in the Streamlit app"""
    base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="800px"
            style="border: none;">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="Document Q&A System")
    init_session_state()
    
    # Initialize managers
    doc_manager = DocumentManager(api_url=BACKEND_URL)
    chat_interface = ChatInterface(api_url=BACKEND_URL)
    
    # Create main layout
    with st.sidebar:
        st.title("Document Management")
        
        # Upload section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Uploading and processing document..."):
                    if doc_manager.upload_document(uploaded_file):
                        st.success("Document uploaded and processed successfully!")
                        # Refresh document list
                        st.session_state.documents = doc_manager.get_documents()
        
        # Document list section
        st.subheader("Available Documents")
        # Refresh button
        if st.button("🔄 Refresh List"):
            st.session_state.documents = doc_manager.get_documents()
        
        # Display documents
        for doc in st.session_state.documents:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                doc_name = doc['filename']
                if st.button(f"📄 {doc_name}", key=f"btn_{doc_name}"):
                    st.session_state.selected_document = doc
            with col2:
                st.write(f"{doc['size'] // 1024}KB")
            with col3:
                st.write("📋")
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Chat", "Document Viewer"])
    
    # Chat tab
    with tab1:
        st.title("Document Q&A Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "source_documents" in message:
                    with st.expander("View Sources"):
                        for source in message["source_documents"]:
                            st.markdown(f"- {source}")
        
        # Chat input
        if question := st.chat_input("Ask a question about your documents"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Get and display response
            with st.spinner("Thinking..."):
                response = chat_interface.get_response(question)
                
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "source_documents": response.get("source_documents", [])
            })
            
            # Force refresh
            st.rerun()
    
    # Document viewer tab
    with tab2:
        st.title("Document Viewer")
        if st.session_state.selected_document:
            doc = st.session_state.selected_document
            st.subheader(f"Viewing: {doc['filename']}")
            
            # Fetch and display PDF content
            pdf_content = doc_manager.get_document_content(doc['filename'])
            if pdf_content:
                display_pdf(pdf_content)
        else:
            st.info("Select a document from the sidebar to view it here")

if __name__ == "__main__":
    main()