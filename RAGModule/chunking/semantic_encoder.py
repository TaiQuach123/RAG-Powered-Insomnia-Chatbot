from typing import Any, Coroutine, List, Optional
from pydantic import BaseModel, Field, validator, PrivateAttr
from transformers import AutoModel
import numpy as np


class BaseEncoder(BaseModel):
    name: str
    score_threshold: Optional[float] = None
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    @validator("score_threshold", pre=True, always=True)
    def set_score_threshold(cls, v):
        return float(v) if v is not None else None

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")

    def acall(self, docs: List[Any]) -> Coroutine[Any, Any, List[List[float]]]:
        raise NotImplementedError("Subclasses must implement this method")

class JinaV3Encoder(BaseEncoder):
    type: str = "JinaEmbeddingModel"
    name: str = "jinaai/jina-embeddings-v3"
    device: str = "cuda"
    max_length: int = 8192
    task: str = "text-matching"
    cache_dir: Optional[str] = None
    threads: Optional[str] = None
    _client: Any = PrivateAttr()
    def __init__(self, score_threshold: float = 0.5, **data):
        super().__init__(score_threshold=score_threshold, **data)
        self._client = self._initialize_client()
    
    def _initialize_client(self):
        embedding = AutoModel.from_pretrained(self.name, trust_remote_code=True).to(self.device)
        embedding.eval()
        return embedding

    def __call__(self, docs: List[str]) -> List[List[float]]:
        try:
            embeds: List[np.ndarray] = list(self._client.encode(docs, task=self.task))
            embeddings: List[List[float]] = [e.tolist() for e in embeds]
            return embeddings
        except Exception as e:
            raise ValueError(f"Failed while Embedding. Error: {e}") from e