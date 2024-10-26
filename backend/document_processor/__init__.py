from .naive_processor import NaiveProcessor

def get_document_processor(processor_type: str="naive", db_name: str="chatbot", collection_name: str="test"):
    if processor_type == "naive":
        return NaiveProcessor(db_name=db_name, collection_name=collection_name)
    else:
        raise ValueError(f"Document processor of type: {processor_type} not implemented!")
