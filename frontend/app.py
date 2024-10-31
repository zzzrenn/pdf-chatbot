import base64
import os
from typing import Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv
from utils.logger import setup_logger

# Environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

# Setup logger
logger = setup_logger("frontend")


def init_session_state(doc_manager):
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_document" not in st.session_state:
        st.session_state.selected_document = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    st.session_state.documents = doc_manager.get_documents()


class DocumentManager:
    def __init__(self, api_url: str = "http://localhost:8000"):
        logger.info(f"DocumentManager initialized with backend API URL: {api_url}")
        self.api_url = api_url

    def upload_document(self, file) -> bool:
        """Upload a document to the backend"""
        logger.info(f"Attempting to upload document: {file.name}")
        try:
            files = {"file": (file.name, file, "application/pdf")}
            response = requests.post(f"{self.api_url}/upload", files=files)
            success = response.status_code == 200
            if success:
                logger.info(f"Successfully uploaded document: {file.name}")
            else:
                logger.error(
                    f"Failed to upload document: {file.name}, Status: {response.status_code}"
                )
            # Update document list
            st.session_state.documents = self.get_documents()
            return success
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            st.error(f"Error uploading document: {str(e)}")
            return False

    def get_documents(self) -> List[Dict]:
        """Fetch list of documents from backend"""
        logger.debug("Fetching document list")
        response = requests.get(f"{self.api_url}/documents")
        if response.status_code == 200:
            documents = response.json()
            logger.info(f"Successfully fetched {len(documents)} documents")
            return documents
        logger.warning(f"Failed to fetch documents, Status: {response.status_code}")
        st.error(f"Error fetching documents: ERROR {response.status_code}")
        return []

    def get_document_content(self, filename: str) -> bytes:
        """Fetch document content from backend"""
        logger.info(f"Fetching content for document: {filename}")
        try:
            response = requests.get(f"{self.api_url}/document/{filename}")
            if response.status_code == 200:
                logger.debug(f"Successfully fetched content for: {filename}")
                return response.content
            logger.warning(f"Failed to fetch document content, Status: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching document content: {str(e)}")
            st.error(f"Error fetching document content: {str(e)}")
            return None


class ChatInterface:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        logger.info(f"ChatInterface initialized with backend API URL: {api_url}")

    def get_response(self, question: str) -> Dict:
        """Get response from chatbot"""
        logger.info(f"Getting response for question: {question}")
        try:
            response = requests.post(f"{self.api_url}/chat", json={"question": question})
            if response.status_code == 200:
                logger.debug("Successfully received chat response")
                return response.json()
            logger.warning(f"Failed to get chat response, Status: {response.status_code}")
            return {
                "answer": "Error: Could not get response",
                "source_documents": [],
            }
        except Exception as e:
            logger.error(f"Error getting chat response: {str(e)}")
            st.error(f"Error getting response: {str(e)}")
            return {
                "answer": "Error: Could not get response",
                "source_documents": [],
            }


def display_pdf(pdf_content: bytes):
    """Display PDF content in the Streamlit app"""
    logger.debug("Displaying PDF content")
    base64_pdf = base64.b64encode(pdf_content).decode("utf-8")
    pdf_display = f"""
        <iframe
            src='data:application/pdf;base64,{base64_pdf}'
            width='100%'
            height='800px'
            style='border: none;'>
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)
    logger.info("Successfully displayed PDF content")


def main():
    logger.info("Starting Streamlit application")
    doc_manager = DocumentManager(api_url=BACKEND_URL)
    chat_interface = ChatInterface(api_url=BACKEND_URL)
    st.set_page_config(layout="wide", page_title="Document Q&A System")
    logger.debug("Initializing session state")
    init_session_state(doc_manager)

    # Create main layout
    with st.sidebar:
        st.title("Document Management")

        # Upload section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file:
            logger.info(f"File selected for upload: {uploaded_file.name}")
            if st.button("Process Document"):
                with st.spinner("Uploading and processing document..."):
                    if doc_manager.upload_document(uploaded_file):
                        logger.info("Document processed successfully")
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
                doc_name = doc["filename"]
                if st.button(f"📄 {doc_name}", key=f"btn_{doc_name}"):
                    st.session_state.selected_document = doc
            with col2:
                st.write(f"{doc['size'] // 1024}KB")
            with col3:
                st.write("📋")

    # Main content area
    col1, col2 = st.columns([6, 4])
    with col1:
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
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "source_documents": response.get("source_documents", []),
                }
            )

            # Force refresh
            st.rerun()

    # Document viewer tab
    with col2:
        st.title("Document Viewer")
        if st.session_state.selected_document:
            doc = st.session_state.selected_document
            st.subheader(f"Viewing: {doc['filename']}")

            # Fetch and display PDF content
            pdf_content = doc_manager.get_document_content(doc["filename"])
            if pdf_content:
                display_pdf(pdf_content)
                logger.info(f"Successfully displayed document: {doc['filename']}")
        else:
            st.info("Select a document from the sidebar to view it here")


if __name__ == "__main__":
    main()
