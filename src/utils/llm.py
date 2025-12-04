"""LLM utility functions for loading HuggingFace models."""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from src.config import settings

_llm_cache: ChatHuggingFace | None = None


def get_huggingface_llm() -> ChatHuggingFace:
    """Get or create HuggingFace LLM singleton from local model path."""
    global _llm_cache
    if _llm_cache is None:
        model_path = settings.llm_model
        
        llm_pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_path,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 1024,
                "temperature": 0.1,
                "do_sample": True,
                "return_full_text": False,
            },
            model_kwargs={
                "trust_remote_code": True,
                "device_map": "auto", 
                
            }
        )
        
        _llm_cache = ChatHuggingFace(llm=llm_pipeline)
        
        print(f"✓ HuggingFace model loaded successfully")
    
    return _llm_cache

