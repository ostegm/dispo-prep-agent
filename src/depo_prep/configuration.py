import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from dataclasses import dataclass
from pydantic import BaseModel, Field

DEFAULT_DEPOSITION_STRUCTURE = """The deposition structure should focus on gathering testimony about the topic:

1. Background/Competency (no document research needed)
   - Establish witness identity and competency
   - Basic background information

2. Main Deposition Topics:
   - Each topic should focus on a key area of testimony
   - Build foundation before key admissions
   - Include document references where applicable
   
3. Wrap-up
   - Catch-all questions
   - Lock in key admissions
   - Allow for cleanup questions"""

class Configuration(BaseModel):
    """Configuration for the deposition preparation process."""
    
    # Model configurations
    planner_model: str = "gemini-2.0-flash-thinking-exp-01-21"
    writer_model: str = "claude-3-sonnet-20240229"
    
    # Planning configurations
    max_sections: int = 3
    max_tokens_per_source: int = 1000
    deposition_organization: str = DEFAULT_DEPOSITION_STRUCTURE
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
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})