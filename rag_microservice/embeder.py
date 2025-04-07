from vertexai.language_models import TextEmbeddingModel

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
        self.__model = TextEmbeddingModel.from_pretrained(embedding_model)
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
        return self.__model.get_embeddings(
            document_chunks_batch, output_dimensionality=self.__dimension
        )

    def embed_query(self, question: str) -> list[float]:
        return self.__model.get_embeddings(
            [self.preprocess_text(question)], output_dimensionality=self.__dimension
        )[0].values
