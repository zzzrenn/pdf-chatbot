# backend/api.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from chatbot import Chatbot
from document_processor import DocumentProcessor
import os
from typing import Dict, List
import shutil
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

app = FastAPI()

# Load env variable
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
DB_NAME = os.getenv("DB_NAME")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

# Initialize chatbot
chatbot = Chatbot(COLLECTION_NAME)

# Initialize document processor
doc_processor = DocumentProcessor(db_name=DB_NAME, collection_name=COLLECTION_NAME)

class Question(BaseModel):
    question: str

@app.post("/chat")
async def chat(question: Question) -> Dict:
    try:
        response = chatbot.get_response(question.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents() -> List[Dict]:
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
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{filename}")
async def get_document(filename: str):
    try:
        file_path = os.path.join(DOCUMENT_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
        return FileResponse(file_path, media_type='application/pdf')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("Uploaded to temp dir")
        
        # Process and store embeddings
        doc_processor.compute_and_store_embeddings(UPLOAD_DIR)
        print("Done computing embeddings...")

        # Move documents from upload dir to document dir
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            shutil.move(file_path, DOCUMENT_DIR)
        shutil.rmtree(UPLOAD_DIR)
        print("Moved to document dir")
        
        return {"message": "Document uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)