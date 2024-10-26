from document_processor.base_processor import BaseProcessor
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple


class NaiveProcessor(BaseProcessor):
    def __init__(self, db_name: str="chatbot", collection_name: str="nice_guidelines_naive"):
        super(NaiveProcessor, self).__init__(db_name=db_name, collection_name=collection_name)
        
    def load_and_split_documents(self, pdf_dir: str) -> List[Document]:
        loader = PyPDFDirectoryLoader(pdf_dir)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = splitter.split_documents(documents)
        return texts
    
    def compute_and_store_embeddings(self, pdf_dir: str):
        # TODO: Check and skip documents already in database
        texts = self.load_and_split_documents(pdf_dir)

        contents = []
        sources = []
        pages = []
        for doc in texts:
            contents.append(doc.page_content)
            metadata = doc.metadata
            sources.append(metadata["source"])
            pages.append(metadata["page"])

        embeddings = self.embeddings.embed_documents(contents)
        
        # Insert documents and their embeddings
        data=[{"vector": e, "text": t, "source": s, "page": p} for e, t, s, p in zip(embeddings, contents, sources, pages)]
        self.vector_store.insert_data(data)
