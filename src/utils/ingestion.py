"""Knowledge base ingestion utilities for Qdrant vector store."""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import DATA_INPUT_DIR, settings

_embeddings: HuggingFaceEmbeddings | None = None
_qdrant_client: QdrantClient | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create embeddings model singleton."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_qdrant_client() -> QdrantClient:
    """Get or create in-memory Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path="qdrant_storage")
    return _qdrant_client


def load_knowledge_base(file_path: Path | None = None) -> str:
    """Load knowledge base text file."""
    if file_path is None:
        file_path = DATA_INPUT_DIR / "knowledge_base.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        return f.read()


def ingest_knowledge_base(file_path: Path | None = None) -> QdrantVectorStore:
    """Ingest knowledge base into Qdrant vector store."""
    text = load_knowledge_base(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_text(text)

    embeddings = get_embeddings()
    client = get_qdrant_client()

    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in collections:
        client.delete_collection(settings.qdrant_collection)

    sample_embedding = embeddings.embed_query("test")
    vector_size = len(sample_embedding)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

    vector_store.add_texts(chunks)

    print(f"Ingested {len(chunks)} chunks into collection '{settings.qdrant_collection}'")
    return vector_store

