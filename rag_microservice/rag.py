import os
import asyncio

import vertexai

from vertexai.preview.generative_models import GenerativeModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma

import json

from embeder import Embedder
from embedings_mamager import EmbeddingsManager
from prompts import (
    QUESTION_PROMTPT,
    FLASHCARD_PROMPT,
    MULTIPL_CHOICE_QUESTIONS_PROMPT,
)


class RAG:
    def __init__(
        self,
        model_ai: str,
        project_id: str,
        region: str,
        embedding_manager: EmbeddingsManager,
    ) -> None:
        vertexai.init(project=project_id, location=region)
        self.__generative_multimodal_model = GenerativeModel(model_ai)
        self.__embedding_manager = embedding_manager

    def __call__(self, question: str, user_id: int) -> str:
        context = self.__embedding_manager.get_closest_chunks(question, user_id, 4, 0.3)
        response = self.__generative_multimodal_model.generate_content(
            QUESTION_PROMTPT.format(
                question=question,
                context=[chunk.page_content for chunk in context],
            )
        )
        return response.text

    def ingest_pdf(self, filename: str, user_id: int) -> None:

        docs = PyPDFLoader(file_path=filename).load()
        self.__embedding_manager.ingest_pdf(
            docs,
            user_id,
        )

    def delete_pdf_embeddings(self, filename: str, user_id: int) -> None:
        self.__embedding_manager.delete_pdf_embeddings(filename, user_id)

    def get_flashcards(
        self, scenario: str, user_id: int, number_of_flashcards: int
    ) -> str:
        context = self.__embedding_manager.get_closest_chunks(scenario, user_id, 4, 0.3)
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
        context = self.__embedding_manager.get_closest_chunks(scenario, user_id, 4, 0.3)
        response = self.__generative_multimodal_model.generate_content(
            MULTIPL_CHOICE_QUESTIONS_PROMPT.format(
                context=[chunk.page_content for chunk in context],
                number_of_questions=number_of_questions,
            )
        )
        return json.loads("\n".join(response.text.split("\n")[1:-1]))
