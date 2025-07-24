from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod

from rag_microservice.embeder.embeder import Embedder


class EmbeddingsManager(ABC):
    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.__embedder = embedder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    @abstractmethod
    def get_closest_chunks(
        self,
        query: str,
        user_id: int,
        number_of_chunks: int,
        similarity_treshold: float,
    ) -> list:
        pass

    @abstractmethod
    def ingest_pdf(self, file: PyPDFLoader, user_id: int) -> None:
        pass

    @abstractmethod
    def delete_pdf_embeddings(self, filename: str, user_id: int) -> None:
        pass
