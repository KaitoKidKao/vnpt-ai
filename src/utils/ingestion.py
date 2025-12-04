"""Knowledge base ingestion utilities for Qdrant vector store."""

import json
import sys
from pathlib import Path

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import DATA_INPUT_DIR, settings

_embeddings: HuggingFaceEmbeddings | None = None
_qdrant_client: QdrantClient | None = None
_vector_store: QdrantVectorStore | None = None

def get_device() -> str:
    """Detect optimal device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create embeddings model singleton."""
    global _embeddings
    if _embeddings is None:
        device = get_device()
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

def get_qdrant_client() -> QdrantClient:
    """Get or create persistent Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        db_path = settings.vector_db_path_resolved
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(db_path))
    return _qdrant_client

def get_vector_store() -> QdrantVectorStore:
    """Get the global vector store instance (Lazy load)."""
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        embeddings = get_embeddings()
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=embeddings,
        )
    return _vector_store


def _initialize_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    force_recreate: bool = False,
) -> None:
    """Initialize Qdrant collection, creating it if needed.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        vector_size: Size of the embedding vectors
        force_recreate: If True, delete existing collection before creating
    """
    collection_exists = client.collection_exists(collection_name)
    
    if collection_exists and force_recreate:
        client.delete_collection(collection_name)
        collection_exists = False
    
    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def load_knowledge_base(file_path: Path | None = None) -> str:
    """Load knowledge base text file."""
    if file_path is None:
        file_path = DATA_INPUT_DIR / "knowledge_base.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        return f.read()

def ingest_knowledge_base(file_path: Path | None = None, force: bool = False) -> QdrantVectorStore:
    """Ingest knowledge base and update singleton."""
    global _vector_store
    
    embeddings = get_embeddings()
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection

    collection_exists = client.collection_exists(collection_name)

    if collection_exists and not force:
        print(f"[Ingestion] Loading existing vector store from: {settings.vector_db_path_resolved}")
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return _vector_store

    if force and collection_exists:
        print(f"[Ingestion] Force re-ingesting: deleting existing collection '{collection_name}'")

    print(f"[Ingestion] Ingesting knowledge base into collection '{collection_name}'...")
    text = load_knowledge_base(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)

    sample_embedding = embeddings.embed_query("test")
    _initialize_collection(client, collection_name, len(sample_embedding), force_recreate=force)

    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    _vector_store.add_texts(chunks, batch_size=64)
    print(f"[Ingestion] Ingested {len(chunks)} chunks into collection '{collection_name}'")
    
    return _vector_store


def ingest_from_crawled_data(
    json_path: Path | str,
    collection_name: str | None = None,
    append: bool = False,
) -> QdrantVectorStore:
    """Ingest crawled JSON data into Qdrant vector store.

    Args:
        json_path: Path to crawled JSON file.
        collection_name: Optional collection name. If None, uses settings.
        append: If True, append to existing collection. If False, recreate.

    Returns:
        QdrantVectorStore instance.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Crawled data not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    if not documents:
        print(f"[Warning] No documents found in {json_path}")
        coll_name = collection_name or settings.qdrant_collection
        embeddings = get_embeddings()
        client = get_qdrant_client()
        return QdrantVectorStore(
            client=client,
            collection_name=coll_name,
            embedding=embeddings,
        )

    texts = []
    metadatas = []
    for doc in documents:
        content = doc.get("content", "")
        if not content:
            continue

        keywords_raw = doc.get("keywords")
        keywords_str = ""
        if isinstance(keywords_raw, list):
            keywords_str = ",".join([str(k) for k in keywords_raw if k])
        elif isinstance(keywords_raw, str):
            keywords_str = keywords_raw
            
        texts.append(content)
        metadatas.append({
            "source_url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "summary": doc.get("summary", ""),
            "topic": data.get("topic", ""),
            "keywords": keywords_str,
            "domain": data.get("domain", ""),
        })

    if not texts:
        raise ValueError(f"No content found in documents from {json_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks = []
    all_metadatas = []
    for text, metadata in zip(texts, metadatas):
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            all_metadatas.append(chunk_metadata)

    embeddings = get_embeddings()
    client = get_qdrant_client()

    coll_name = collection_name or settings.qdrant_collection
    collection_exists = client.collection_exists(coll_name)

    if not append or not collection_exists:
        sample_embedding = embeddings.embed_query("test")
        _initialize_collection(client, coll_name, len(sample_embedding), force_recreate=not append)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=coll_name,
        embedding=embeddings,
    )

    vector_store.add_texts(all_chunks, metadatas=all_metadatas)

    print(f"[Ingestion] Ingested {len(all_chunks)} chunks from {len(documents)} documents")
    print(f"[Ingestion] Collection: '{coll_name}'")
    print(f"[Ingestion] Source: {data.get('source', json_path)}")
    return vector_store


def load_pdf(file_path: Path) -> str:
    """Load text from PDF file.
    
    Raises:
        ImportError: If pypdf is not installed
        Exception: If file cannot be read
    """
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is required for PDF files. Install with: pip install pypdf")
    
    try:
        reader = pypdf.PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF file {file_path}: {e}") from e


def load_docx(file_path: Path) -> str:
    """Load text from DOCX file.
    
    Raises:
        ImportError: If python-docx is not installed
        Exception: If file cannot be read
    """
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx is required for DOCX files. Install with: pip install python-docx")
    
    try:
        doc = docx.Document(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise RuntimeError(f"Failed to read DOCX file {file_path}: {e}") from e


def load_txt(file_path: Path) -> str:
    """Load text from TXT file.
    
    Raises:
        IOError: If file cannot be read
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise IOError(f"Failed to read TXT file {file_path}: {e}") from e


def load_file(file_path: Path) -> tuple[str | None, dict | None]:
    """Load file and return (text, metadata).
    
    Args:
        file_path: Path to file (PDF, DOCX, TXT, or JSON)
        
    Returns:
        Tuple of (text, metadata) or (None, None) for JSON/unsupported files
        
    Raises:
        ImportError: If required library is missing for PDF/DOCX
        IOError: If file cannot be read
    """
    ext = file_path.suffix.lower()
    
    if ext == ".json":
        return None, None
    
    try:
        if ext == ".pdf":
            text = load_pdf(file_path)
        elif ext == ".docx":
            text = load_docx(file_path)
        elif ext == ".txt":
            text = load_txt(file_path)
        else:
            print(f"[Warning] Unsupported file type: {ext}")
            return None, None
        
        metadata = {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "file_type": ext[1:],
        }
        return text, metadata
    except (ImportError, IOError, RuntimeError) as e:
        print(f"[Error] Failed to load {file_path.name}: {e}")
        return None, None


def ingest_files(
    file_paths: list[Path],
    collection_name: str | None = None,
    append: bool = False,
) -> int:
    """Ingest multiple files (PDF, DOCX, TXT, JSON) into Qdrant.
    
    Args:
        file_paths: List of file paths to ingest
        collection_name: Optional collection name. If None, uses settings.
        append: If True, append to existing collection. If False, recreate.
        
    Returns:
        Total number of chunks ingested
    """
    embeddings = get_embeddings()
    client = get_qdrant_client()
    
    coll_name = collection_name or settings.qdrant_collection
    collection_exists = client.collection_exists(coll_name)
    
    if not append or not collection_exists:
        sample_embedding = embeddings.embed_query("test")
        _initialize_collection(client, coll_name, len(sample_embedding), force_recreate=not append)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=coll_name,
        embedding=embeddings,
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    
    total_chunks = 0
    total_docs = 0
    
    for file_path in file_paths:
        if file_path.suffix.lower() == ".json":
            try:
                ingest_from_crawled_data(file_path, coll_name, append=True)
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                total_docs += len(data.get("documents", []))
                print(f"  [OK] {file_path.name}")
            except Exception as e:
                print(f"  [Error] {file_path.name}: {e}")
            continue
        
        text, metadata = load_file(file_path)
        if not text:
            continue
        
        chunks = splitter.split_text(text)
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(chunks)
            metadatas.append(chunk_meta)
        
        vector_store.add_texts(chunks, metadatas=metadatas)
        total_chunks += len(chunks)
        total_docs += 1
        print(f"  [OK] {file_path.name} ({len(chunks)} chunks)")
    
    print(f"\n[Ingest] Total: {total_docs} documents, {total_chunks} chunks")
    print(f"[Ingest] Collection: '{coll_name}'")
    return total_chunks
