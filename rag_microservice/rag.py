import os
import asyncio

import vertexai

from vertexai.preview.generative_models import GenerativeModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import json

from embeder import Embedder
from prompts import (
    QUESTION_PROMTPT,
    FLASHCARD_PROMPT,
    MULTIPL_CHOICE_QUESTIONS_PROMPT,
)


class RAG:
    def __init__(self, model_ai: str, project_id: str, region: str) -> None:
        vertexai.init(project=project_id, location=region)
        self.__generative_multimodal_model = GenerativeModel(model_ai)
        self.embedding_model = Embedder()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048, chunk_overlap=100
        )
        self.__retriever = None

    def __call__(self, question: str, user_id: int) -> str:
        context = self.most_suiting_pdf(question, user_id)
        response = self.__generative_multimodal_model.generate_content(
            QUESTION_PROMTPT.format(
                question=question,
                context=[chunk.page_content for chunk in context],
            )
        )
        return response.text

    def ingest_pdf(self, filename: str, user_id: int) -> None:
        vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=f"chroma_db/{user_id}",
        )
        docs = PyPDFLoader(file_path=filename).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(filename)
        vector_store.add_documents(chunks)
        vector_store.persist()

    def most_suiting_pdf(self, query: str, user_id: int) -> list:
        vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=f"chroma_db/{user_id}",
        )
        if self.__retriever is None:
            self.__retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 4,
                    "score_threshold": 0.3,
                },
            )

        return self.__retriever.invoke(query)

    def delete_pdf_embeddings(self, filename: str, user_id: int) -> None:
        collection = Chroma(
            embedding_function=self.embedding_model,
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

    def get_flashcards(
        self, scenario: str, user_id: int, number_of_flashcards: int
    ) -> str:
        context = self.most_suiting_pdf(scenario, user_id)
        response = self.__generative_multimodal_model.generate_content(
            FLASHCARD_PROMPT.format(
                context=[chunk.page_content for chunk in context],
                number_of_flashcards=number_of_flashcards,
            )
        )
        return json.loads("\n".join(response.text.split("\n")[1:-1]))

    def get_multiple_choice_questions(
        self, scenario: str, user_id: int, number_of_questions: int
    ) -> str:
        context = self.most_suiting_pdf(scenario, user_id)
        response = self.__generative_multimodal_model.generate_content(
            MULTIPL_CHOICE_QUESTIONS_PROMPT.format(
                context=[chunk.page_content for chunk in context],
                number_of_questions=number_of_questions,
            )
        )
        return json.loads("\n".join(response.text.split("\n")[1:-1]))


# async def main():
#     load_dotenv()
#     _AR_PROJECT_ID = os.getenv("_AR_PROJECT_ID")
#     _DEPLOY_REGION = os.getenv("_DEPLOY_REGION")

#     rag = RAG(
#         model_ai=os.getenv("_AI_CHATBOT"),
#         project_id=_AR_PROJECT_ID,
#         region=_DEPLOY_REGION,
#     )

#     file = "/home/mikic202/hackathon/Backend/backend/rag/1_PDF_chapter_1.pdf"
#     # file = "/home/mikic202/hackathon/Backend/backend/rag/2_PDF_chapters_1_to_5.pdf"
#     # file = "/home/mikic202/hackathon/Backend/backend/rag/3_PDF_full.pdf"
#     # taks = asyncio.create_task(rag.ingest_pdf(file, 1))
#     rag.ingest_pdf(file, 1)
#     # while not taks.done():
#     #     print("Ingesting PDF...")
#     #     await asyncio.sleep(1)
#     question = "When was linux introduced?"
#     # qt = asyncio.create_task(rag.get_multiple_choice_questions(question, 1, 5))
#     # qt = asyncio.create_task(rag.get_flashcards(question, 1, 5))
#     qt = asyncio.create_task(rag(question, 1))
#     print("___________________________________")
#     while not qt.done():
#         print(f"Progress: {1}")
#         await asyncio.sleep(1)
#     print(f"Progress: {qt.result()}")

#     # print(a.result())


# if __name__ == "__main__":
#     asyncio.run(main())
