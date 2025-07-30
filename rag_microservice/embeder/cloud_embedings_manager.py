from embeder.embedings_mamager import EmbeddingsManager
from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine
from langchain.embeddings import VertexAIEmbeddings
import os

from rag_microservice.embeder.embeder import Embedder


class EmbedingManager(EmbeddingsManager):
    def __init__(
        self, embedder: Embedder, chunk_size: int, chunk_overlap: int, database: str
    ) -> None:
        super().__init__(embedder, chunk_size, chunk_overlap)
        self.__engine = PostgresEngine.from_instance(
            os.environ.get("PROJECT_ID"),
            os.environ.get("REGION"),
            os.environ.get("INSTANCE"),
            database,
        )

    def get_closest_chunks(
        self,
        query: str,
        user_id: int,
        number_of_chunks: int,
        similarity_treshold: float,
    ) -> list:
        vectorstore = PostgresVectorStore.create_sync(
            engine,
            table_name=f"user_{user_id}",
            embedding_service=self.__embedder,
            metadata_columns=["source"],
        )
        retireiver = vectorstore.as_retriever(
            k=number_of_chunks, score_threshold=similarity_treshold
        )
        return retireiver.invoke(query)

    def ingest_pdf(self, file, user_id: int) -> None:
        vectorstore = PostgresVectorStore.create_sync(
            self.__engine,
            table_name=f"user_{user_id}",
            embedding_service=self.__embedder,
            metadata_columns=["source"],
        )
        chunks = self.text_splitter.split_documents(file.load())
        vectorstore.add_documents(
            chunks,
            metadatas=[
                {"source": os.path.basename(file.source)} for _ in range(len(chunks))
            ],
        )

    def delete_pdf_embeddings(self, filename: str, user_id: int) -> None:
        vectorstore = PostgresVectorStore.create_sync(
            self.__engine,
            table_name=f"user_{user_id}",
            embedding_service=self.__embedder,
            metadata_columns=["source"],
        )
        vectorstore.delete(filter={"source": os.path.basename(filename)})
