import os
from typing import Dict

from document_processor import get_document_processor
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")


class Chatbot:
    def __init__(
        self,
        db_name: str,
        collection_name: str,
        vector_store_uri: str = "http://127.0.0.1:19530",
        doc_processor_type: str = "naive",
        bm25=True,
        rerank=True,
    ):
        self.collection_name = collection_name
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.vector_store = Milvus(
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name=collection_name,
            vector_field="vector",
            text_field="text",
            connection_args={"uri": vector_store_uri, "db_name": db_name},
        )
        self.doc_processor = get_document_processor(
            processor_type=doc_processor_type,
            db_name=db_name,
            collection_name=collection_name,
        )
        self.bm25 = bm25
        self.reranker = CohereRerank(model="rerank-english-v3.0", top_n=3) if rerank else None
        self.chat_history = []
        self._create_chain()

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
        semantic_retriever = self.vector_store.as_retriever(kwargs={"k": 5})

        if self.bm25:
            # BM25
            chunks = self.doc_processor.load_and_split_documents(DOCUMENT_DIR)
            if len(chunks) == 0:
                Warning("No documents in database...")
                retriever = semantic_retriever
            else:
                bm25_retriever = BM25Retriever.from_documents(chunks, kwargs={"k": 5})

                # initialize the ensemble retriever with 3 Retrievers
                retriever = EnsembleRetriever(
                    retrievers=[semantic_retriever, bm25_retriever],
                    weights=[0.8, 0.2],
                )

        if self.reranker:
            # reranking
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker, base_retriever=retriever
            )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, compression_retriever, contextualize_q_prompt
        )

        # Define the prompt to answer the question
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the question. "
            "If you are unsure about the question, ask the user for clarification. "
            "If you don't know the answer, say you don't know."
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
        self.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_response(self, query: str) -> Dict:
        response = self.chain.invoke({"input": query, "chat_history": self.chat_history})
        sources = set()
        for context in response["context"]:
            metadata = context.metadata
            sources.add((metadata["source"], "page: " + str(metadata["page"])))
        self.chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=response["answer"]),
            ]
        )
        return {
            "answer": response["answer"],
            "source_documents": list(sources),
        }


if __name__ == "__main__":
    chatbot = Chatbot(db_name="chatbot", collection_name="nice_guidelines")
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
