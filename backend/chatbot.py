# backend/chatbot.py
from langchain_openai  import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_milvus import Milvus
from typing import Dict

class Chatbot:
    def __init__(self, collection_name: str, vector_store_uri: str="http://127.0.0.1:19530"):
        self.collection_name = collection_name
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini"
        )
        self.vector_store = Milvus(
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name=collection_name,
            vector_field="vector",
            text_field="text",
            connection_args={"uri": vector_store_uri},
        )
        self.memory = ConversationBufferMemory(
            output_key="answer",
            memory_key="chat_history",
            return_messages=True
        )
        self.chain = self._create_chain()
    
    def _create_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True
        )
    
    def get_response(self, question: str) -> Dict:
        response = self.chain({"question": question})
        return {
            "answer": response["answer"],
            "source_documents": response["source_documents"]
        }
    
if __name__ == "__main__":
    chatbot = Chatbot(collection_name="nice_guidelines")
    print("chatbot ready...")
    question = "what is the first-line treatment for pregnant women with hypertension?"
    res = chatbot.get_response(question=question)
    print(res["answer"])
    print(res["source_documents"])
    print("chatbot test ok!")