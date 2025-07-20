from flask import Flask, Response
from rag import RAG
from embedings_mamager import EmbeddingsManager
from embeder import Embedder
import vertexai


class Gateway(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_url_rule("hello", "hello", self.hello, methods=["GET"])
        self.add_url_rule("basic", "basic", self.get_basic_llm_answer, methods=["GET"])
        self.add_url_rule(
            "rag",
            "rag",
            self.get_rag_answer,
            methods=["GET"],
        )
        self.add_url_rule(
            "add_user_file",
            "add_user_file",
            self.add_user_file,
            methods=["POST"],
        )
        vertexai.init(project="basic-rag-456110", location="europe-west1")
        self.__embedder = Embedder("text-embedding-005", 512)
        self.__embedding_manager = EmbeddingsManager(
            "chroma_db", self.__embedder, 4096, 100
        )

        self.__rag = RAG(
            "gemini-2.0-flash-001",
            self.__embedding_manager,
            4,
            0.3,
        )

    def hello(self):
        return Response("Hello, World!", status=200, mimetype="text/plain")

    def get_basic_llm_answer(self, question: str) -> str:
        return f"Basic answer to your question: {question}"

    def get_rag_answer(self, question: str, user_id: int) -> str:
        return f"RAG answer to your question: {question} for user {user_id}"

    def add_user_file(self, user_id: int, file: str) -> str:
        return f"File {file} added for user {user_id}"
