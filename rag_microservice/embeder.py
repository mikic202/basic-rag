
from google import genai
from google.genai.types import EmbedContentConfig

from google.genai.types import HttpOptions

from request_controller import RequestController

SAFE_MARGIN = 18000


class Embedder:
    DOCUMENT_RETRIEVAL_TASK_TYPE = "RETRIEVAL_DOCUMENT"
    QUERY_RETRIEVAL_TASK_TYPE = "RETRIEVAL_QUERY"

    def __init__(
        self,
        embedding_model: str,
        embedding_dimension: int,
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
        number_of_tokens_to_embede = 0
        for chunk in documents:
            chunk_tokens_number = self.count_tokens_estimate(chunk)

            if number_of_tokens_to_embede + chunk_tokens_number > SAFE_MARGIN:
                if document_chunks_batch:
                    self.request_controller.attempt_request()
                    response += self.embed_document_chunks(document_chunks_batch)
                    document_chunks_batch = []

            document_chunks_batch.append(self.preprocess_text(chunk))
            number_of_tokens_to_embede += chunk_tokens_number

        if document_chunks_batch:
            self.request_controller.attempt_request()
            response += self.embed_document_chunks(document_chunks_batch)
        return [embedding.values for embedding in response]

    def embed_document_chunks(self, document_chunks_batch: list) -> list[list[float]]:
        return self.client.models.embed_content(
            model=self.model,
            contents=document_chunks_batch,
            config=EmbedContentConfig(
                task_type=self.DOCUMENT_RETRIEVAL_TASK_TYPE,
                output_dimensionality=self.__dimension,
            ),
        ).embeddings

    def embed_query(self, question: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=[question],
            config=EmbedContentConfig(
                task_type=self.QUERY_RETRIEVAL_TASK_TYPE,
                output_dimensionality=self.__dimension,
            ),
        )
        return response.embeddings[0].values
