# backend/chatbot.py
from langchain_openai  import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
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
        self.chat_history = []
        self.chain = self._create_chain()
    
    def _create_chain(self):
        # Define the prompt to contextualize the question
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        retriever = self.vector_store.as_retriever(kwargs={"k":5})
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Define the prompt to answer the question
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create the question-answering chain
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # Create the retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return rag_chain
            
    def get_response(self, query: str) -> Dict:
        response = self.chain.invoke({"input": query, "chat_history": self.chat_history})
        self.chat_history.extend([HumanMessage(content=query), AIMessage(content=response["answer"])])
        return {
            "answer": response["answer"],
            "source_documents": []
        }
    
if __name__ == "__main__":
    chatbot = Chatbot(collection_name="nice_guidelines")
    print("chatbot ready...")
    question = "what is hypertension?"
    res = chatbot.get_response(query=question)
    print(res["answer"])
    print(res["source_documents"])

    question = "what is first line treatment for pregnant women?"
    res = chatbot.get_response(query=question)
    print(res["answer"])
    print(res["source_documents"])
    print("chatbot test ok!")