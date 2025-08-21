from flask import Flask, Response, request, redirect, flash
from rag.rag import RAG
from embeder.local_embedings_manager import LocalEmbeddingsManager
from embeder.cloud_embedings_manager import CloudEmbedingManager
from embeder.embeder import Embedder
from werkzeug.utils import secure_filename
from pathlib import Path
import vertexai
import os


class Gateway(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_url_rule(
            "/basic", "/basic", self.get_basic_llm_answer, methods=["GET"]
        )
        self.add_url_rule(
            "/rag",
            "/rag",
            self.get_rag_answer,
            methods=["GET"],
        )
        self.add_url_rule(
            "/add_user_file/<user_id>",
            "/add_user_file/<user_id>",
            self.add_user_file,
            methods=["POST"],
        )
        self.__file_directory = Path("user_files")
        self.__init_models_form_env()
        self.__number_of_chunks = 4
        self.__similarity_treshold = 0.3

    def __init_models_form_env(self):
        if os.environ.get("EMBEDDER_TYPE") == "cloud":
            self.__embedder = Embedder(
                os.environ.get("EMBEDDER_MODEL"),
                int(os.environ.get("EMBEDDER_DIMENSIONS")),
            )
        else:
            pass

        if os.environ.get("EMBEDDINGS_MANAGER_TYPE") == "cloud":
            self.__embedding_manager = CloudEmbedingManager(
                os.environ.get("EMBEDDINGS_STROE_PATH"),
                self.__embedder,
                int(os.environ.get("CHUNK_SIZE", default=4096)),
                int(os.environ.get("CHUNK_OVERLAP", default=100)),
            )
        else:
            self.__embedding_manager = LocalEmbeddingsManager(
                os.environ.get("EMBEDDINGS_STROE_PATH"),
                self.__embedder,
                int(os.environ.get("CHUNK_SIZE", default=4096)),
                int(os.environ.get("CHUNK_OVERLAP", default=100)),
            )

        if os.environ.get("LLM_TYPE") == "cloud":
            vertexai.init(
                project=os.environ.get("PROJECT_ID"),
                location=os.environ.get("REGION"),
            )
            self.__rag = RAG(
                os.environ.get("LLM_MODEL"),
                self.__embedding_manager,
                4,
                0.3,
            )
        else:
            pass

    def get_basic_llm_answer(self, question: str) -> str:
        return Response(
            self.__rag.get_simple_answer(question), status=200, mimetype="text/plain"
        )

    def get_rag_answer(self, question: str, user_id: int) -> str:
        return Response(
            self.__rag(
                question, user_id, self.__number_of_chunks, self.__similarity_treshold
            ),
            status=200,
            mimetype="text/plain",
        )

    def add_user_file(self, user_id) -> str:
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return Response("No file selected", status=400, mimetype="text/plain")
        filename = secure_filename(file.filename)
        file_directory = self.__file_directory / str(user_id)
        file_directory.mkdir(parents=True, exist_ok=True)
        file.save(file_directory / filename)
        self.__rag.ingest_pdf(str(file_directory / filename), user_id)
        return Response("Successfully uploaded file", status=200, mimetype="text/plain")
