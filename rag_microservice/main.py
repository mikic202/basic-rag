from rag_microservice.rag.rag import RAG
from rag_microservice.embeder.embedings_mamager import EmbeddingsManager
from rag_microservice.embeder.embeder import Embedder
import vertexai


if __name__ == "__main__":
    vertexai.init(project="basic-rag-456110", location="europe-west1")
    embedder = Embedder("text-embedding-005", 512)
    embedding_manager = EmbeddingsManager("chroma_db", embedder, 4096, 100)

    rag = RAG(
        "gemini-2.0-flash-001",
        embedding_manager,
        4,
        0.3,
    )

    file = "1_PDF_chapter_1.pdf"
    # taks = asyncio.create_task(rag.ingest_pdf(file, 1))
    # rag.ingest_pdf(file, 1)
    # while not taks.done():
    #     print("Ingesting PDF...")
    #     await asyncio.sleep(1)

    print("LLLLLLLLLLLLLLLLLLLLLLL")
    question = "When Linux development started?"
    print(rag(question, 1))
