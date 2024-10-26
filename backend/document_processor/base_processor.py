from abc import ABC
from langchain_openai import OpenAIEmbeddings
import os
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../database'))
from typing import List
from database import VectorStore

class BaseProcessor(ABC):
    def __init__(self, db_name: str="chatbot", collection_name: str="nice_guidelines"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = VectorStore(db_name=db_name, collection_name=collection_name)
        
    def load_and_split_documents(self, pdf_dir: str) -> List[str]:
        pass
    
    def compute_and_store_embeddings(self, pdf_dir: str):
        pass
    
