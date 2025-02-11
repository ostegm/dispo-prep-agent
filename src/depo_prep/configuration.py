import os
from dataclasses import fields
from typing import Any, Optional
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

class Configuration(BaseModel):
    """Configuration for the deposition preparation process."""
    
    # Model configurations
    planner_model: str = "gemini-2.0-flash" #-thinking-exp-01-21"
    writer_model: str = "claude-3-sonnet-20240229"
    
    # Planning configurations
    max_sections: int = 3
    max_tokens_per_source: int = 1000
    default_num_queries_per_section: int = 1
    
    # Vector DB configurations
    chroma_persist_dir: str = Field(
        default_factory=lambda: str(Path(__file__).parent.parent.parent / "chroma_db")
    )
    chroma_collection_name: str = "case_documents"
    embedding_model: str = "text-embedding-ada-002"
    max_results_per_query: int = 5

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure Chroma directory exists
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        # Use model_fields instead of dataclass fields
        values = {
            field: configurable.get(field) 
            for field in cls.model_fields.keys()
        }
        return cls(**{k: v for k, v in values.items() if v is not None})