from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

from embeder.embeder import Embedder
from embeder.embedings_mamager import EmbeddingsManager


class LocalEmbeddingsManager(EmbeddingsManager):
    def __init__(
        self,
        store_directory: str,
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.__store_directory = store_directory
        super().__init__(
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def get_closest_chunks(
        self,
        query: str,
        user_id: int,
        number_of_chunks: int,
        similarity_treshold: float,
    ) -> list:
        vector_store = Chroma(
            embedding_function=self.__embedder,
            persist_directory=f"{self.__store_directory}/{user_id}",
        )
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": number_of_chunks,
                "score_threshold": similarity_treshold,
            },
        )
        return retriever.invoke(query)

    def ingest_pdf(self, file: PyPDFLoader, user_id: int) -> None:
        vector_store = Chroma(
            embedding_function=self.__embedder,
            persist_directory=f"{self.__store_directory}/{user_id}",
        )
        chunks = filter_complex_metadata(
            self.text_splitter.split_documents(file.load())
        )
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(file.source)
        vector_store.add_documents(chunks)
        vector_store.persist()

    def delete_pdf_embeddings(self, filename: str, user_id: int) -> None:
        collection = Chroma(
            embedding_function=self.__embedder,
            persist_directory=f"chroma_db/{user_id}",
        )

        all_data = collection.get()

        file_basename = os.path.basename(filename)
        ids_to_delete = [
            _id
            for _id, metadata in zip(all_data["ids"], all_data["metadatas"])
            if metadata.get("source") == file_basename
        ]

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            collection.persist()
