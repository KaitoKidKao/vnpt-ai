"""LLM utility functions for loading HuggingFace models."""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from src.config import settings

_small_llm_cache: ChatHuggingFace | None = None
_large_llm_cache: ChatHuggingFace | None = None


def _load_model(model_path: str, model_type: str) -> ChatHuggingFace:
    """Internal helper to load a HuggingFace model."""
    
    llm_pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_path,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 1024,
            "do_sample": False,
            "return_full_text": False,
        },
        model_kwargs={
            "trust_remote_code": True,
            "device_map": "auto",
            "do_sample": False,
        }
    )
    
    llm = ChatHuggingFace(llm=llm_pipeline)
    print(f"[Model] {model_type} model loaded successfully from {model_path}")
    
    return llm


def get_small_model() -> ChatHuggingFace:
    """Get or create small HuggingFace LLM singleton (for router)."""
    global _small_llm_cache
    if _small_llm_cache is None:
        _small_llm_cache = _load_model(settings.llm_model_small, "Small")
    return _small_llm_cache


def get_large_model() -> ChatHuggingFace:
    """Get or create large HuggingFace LLM singleton (for RAG and logic)."""
    global _large_llm_cache
    if _large_llm_cache is None:
        _large_llm_cache = get_small_model()
    return _large_llm_cache

