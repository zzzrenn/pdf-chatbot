from pymilvus import connections, db, DataType, Collection, utility, FieldSchema, CollectionSchema

class VectorStore:
    def __init__(self, db_name="guideline_chatbot", collection_name="test"):
        conn = connections.connect(host="127.0.0.1", port=19530)
        self.db_name = db_name
        if db_name not in db.list_database():
            print("created database")
            db.create_database(db_name)
        db.using_database(db_name)
        self.collection = self.create_collection(collection_name)

    def create_collection(self, collection_name: str="embeddings", dim: int=1536):
        if not utility.has_collection(collection_name):
            # Field schema
            id = FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            )
            vector = FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim, # default openAI embeddings dim 1536
            )
            text = FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=50000
            )
            source = FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=1000
            )
            page = FieldSchema(
                name="page",
                dtype=DataType.INT32
            )

            # Collection schema
            schema = CollectionSchema(
                fields=[id, vector, text, source, page], 
                description="NICE guidelines' embeddings"
            )

            # create collection
            collection = Collection(
                name=collection_name, 
                schema=schema, 
                using='default'
            )

            # create index
            index_params = {
                "metric_type":"IP",
                "index_type":"FLAT",
            }
            collection.create_index(
                field_name="vector", 
                index_params=index_params
            )
        else:
            collection = Collection(collection_name)
            collection.load()
        return collection
    
    def insert_data(self, data):
        return self.collection.insert(data)
    
    def query(self, query, output_fields):
        self.collection.load()
        res = self.collection.query(
            expr = query, 
            output_fields = output_fields
        )
        return res
if __name__ == "__main__":
    import random
    
    vector = [[random.uniform(-1, 1) for _ in range(1536)]]
    text = ["test text"]
    source = ["dummy"]
    page = [100]
    data = [{"vector": v, "text": t, "source": s, "page": p} for v,t,s,p in zip(vector, text, source, page)]

    db = VectorStore(db_name="HI", collection_name="test")
    res = db.insert_data(data)
    print(res)
    print("added data...")

    # try retrieve data
    res = db.query('source == "dummy"', ["id", "text", "source", "page"])
    print(res)
    print("test ok")
