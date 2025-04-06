import os

from google import genai
from google.genai.types import EmbedContentConfig

from google.genai.types import HttpOptions

from request_controller import RequestController

SAFE_MARGIN = 18000


class Embedder:
    def __init__(
        self,
        embedding_model: str = "text-embedding-005",
        embedding_dimension: int = 512,
    ) -> None:
        self.__dimension = embedding_dimension
        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
        self.model = embedding_model
        self.request_controller = RequestController()

    def count_tokens_estimate(self, text: str) -> int:
        return (len(text) - text.count(" ")) // 4

    def preprocess_text(self, text: str) -> str:
        return text.replace(". .", "").replace("  ", " ")

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        response = []
        document_chunks_batch = []
        current_tokens = 0
        for chunk in documents:
            chunk_tokens = self.count_tokens_estimate(chunk)

            if current_tokens + chunk_tokens > SAFE_MARGIN:
                if document_chunks_batch:
                    self.request_controller.attempt_request()
                    response += self.client.models.embed_content(
                        model=self.model,
                        contents=document_chunks_batch,
                        config=EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=self.__dimension,
                        ),
                    ).embeddings
                    document_chunks_batch = []

            document_chunks_batch.append(self.preprocess_text(chunk))
            current_tokens += chunk_tokens

        if document_chunks_batch:
            self.request_controller.attempt_request()
            response += self.client.models.embed_content(
                model=self.model,
                contents=document_chunks_batch,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.__dimension,
                ),
            ).embeddings
            document_chunks_batch = []
        return [embedding.values for embedding in response]

    def embed_query(self, question: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=[question],
            config=EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.__dimension,
            ),
        )
        return response.embeddings[0].values
