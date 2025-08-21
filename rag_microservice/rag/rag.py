from vertexai.preview.generative_models import GenerativeModel
from langchain_community.document_loaders import PyPDFLoader

import json

from embeder.embedings_mamager import EmbeddingsManager
from rag.prompts import (
    QUESTION_PROMTPT,
    FLASHCARD_PROMPT,
    MULTIPL_CHOICE_QUESTIONS_PROMPT,
)


class RAG:
    def __init__(
        self,
        model_ai: str,
        embedding_manager: EmbeddingsManager,
        chunks_in_context: int,
        chunks_similarity_treshold: float,
    ) -> None:
        self.__generative_multimodal_model = GenerativeModel(model_ai)
        self.__embedding_manager = embedding_manager
        self.__chunks_similarity_treshold = chunks_similarity_treshold
        self.__chunks_in_context = chunks_in_context

    def __call__(self, question: str, user_id: int) -> str:
        context = self.__embedding_manager.get_closest_chunks(
            question,
            user_id,
            self.__chunks_in_context,
            self.__chunks_similarity_treshold,
        )
        response = self.__generative_multimodal_model.generate_content(
            QUESTION_PROMTPT.format(
                question=question,
                context=[chunk.page_content for chunk in context],
            )
        )
        return response.text

    def ingest_pdf(self, filename: str, user_id: int) -> None:
        docs = PyPDFLoader(file_path=filename)
        self.__embedding_manager.ingest_pdf(
            docs,
            user_id,
        )

    def delete_pdf_embeddings(self, filename: str, user_id: int) -> None:
        self.__embedding_manager.delete_pdf_embeddings(filename, user_id)

    def get_flashcards(
        self, scenario: str, user_id: int, number_of_flashcards: int
    ) -> str:
        context = self.__embedding_manager.get_closest_chunks(
            scenario,
            user_id,
            self.__chunks_in_context,
            self.__chunks_similarity_treshold,
        )
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
        context = self.__embedding_manager.get_closest_chunks(
            scenario,
            user_id,
            self.__chunks_in_context,
            self.__chunks_similarity_treshold,
        )
        response = self.__generative_multimodal_model.generate_content(
            MULTIPL_CHOICE_QUESTIONS_PROMPT.format(
                context=[chunk.page_content for chunk in context],
                number_of_questions=number_of_questions,
            )
        )
        return json.loads("\n".join(response.text.split("\n")[1:-1]))

    def get_simple_answer(self, question: str) -> str:
        return self.__generative_multimodal_model.generate_content(question).text
