from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from chatbot import Chatbot
from document_processor import get_document_processor
import os
from typing import Dict, List
import shutil
from dotenv import load_dotenv
from utils.logger import setup_logger
import traceback

# Setup logger
logger = setup_logger("backend_api")

app = FastAPI()

# Load env variable
load_dotenv(dotenv_path=".env")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
DB_NAME = os.getenv("DB_NAME")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")
DOCUMENT_PROCESSOR_TYPE = os.getenv("DOCUMENT_PROCESSOR_TYPE")
BM25 = os.getenv("BM25").lower() == "true"
RERANK = os.getenv("RERANK").lower() == "true"

# Initialize chatbot
logger.debug(f"Initializing chatbot with db_name={DB_NAME}, collection_name={COLLECTION_NAME}, doc_processor_type={DOCUMENT_PROCESSOR_TYPE}, bm25={BM25}, rerank={RERANK}")
try:
    chatbot = Chatbot(DB_NAME, COLLECTION_NAME, doc_processor_type=DOCUMENT_PROCESSOR_TYPE, bm25=BM25, rerank=RERANK)
    logger.info("Chatbot initialized succesfully")
except Exception as e:
    logger.critical(f"Failed to initialize chatbot: {str(e)}")
    raise

# Initialize document processor
logger.debug(f"Initializing document processor with processor_type={DOCUMENT_PROCESSOR_TYPE}, db_name={DB_NAME}, collection_name={COLLECTION_NAME}")
try:
    doc_processor = get_document_processor(processor_type=DOCUMENT_PROCESSOR_TYPE, db_name=DB_NAME, collection_name=COLLECTION_NAME)
    logger.info("Document processor initialized succesfully")
except Exception as e:
    logger.critical(f"Failed to initialize document processor: {str(e)}")
    raise

class Question(BaseModel):
    question: str

@app.post("/chat")
async def chat(question: Question) -> Dict:
    logger.info(f"Received chat request with question: {question.question}")
    try:
        response = chatbot.get_response(question.question)
        logger.debug(f"Chat response generated: {response['answer']}\nSources: {response['source_documents']}...")
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents() -> List[Dict]:
    logger.info("Received request to list documents")
    try:
        documents = []
        for filename in sorted(os.listdir(DOCUMENT_DIR)):
            if filename.endswith('.pdf'):
                file_path = os.path.join(DOCUMENT_DIR, filename)
                documents.append({
                    "filename": filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path)
                })
        logger.debug(f"Found {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error in documents endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{filename}")
async def get_document(filename: str):
    logger.info(f"Received request for document: {filename}")
    try:
        file_path = os.path.join(DOCUMENT_DIR, filename)
        if not os.path.exists(file_path):
            logger.warning(f"Document not found: {filename}")
            raise HTTPException(status_code=404, detail="Document not found")
        return FileResponse(file_path, media_type='application/pdf')
    except Exception as e:
        logger.error(f"Error in document retrieval endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Received upload request for file: {file.filename}")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        logger.debug(f"Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved successfully: {file.filename}")
        
        # Process and store embeddings
        logger.debug("Processing document and computing embeddings")
        doc_processor.compute_and_store_embeddings(UPLOAD_DIR)
        logger.info("Document processed and embeddings stored successfully")

        # Move documents from upload dir to document dir
        logger.debug("Moving processed document to document storage directory")
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            shutil.move(file_path, DOCUMENT_DIR)
        shutil.rmtree(UPLOAD_DIR)
        logger.info("Document moved to storage directory succesfully")

        # update BM25 retriever
        chatbot._create_chain()
        logger.info("Updated chatbot chain succesfully")
        
        return {"message": "Document uploaded and processed successfully"}
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)