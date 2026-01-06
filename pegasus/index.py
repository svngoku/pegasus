"""Vector index management using USearch HNSW."""

from pathlib import Path
from typing import List

import numpy as np
from usearch.index import Index as USearchIndex, MetricKind, Matches


class VectorIndexManager:
    """Manages the USearch HNSW index."""
    
    def __init__(
        self,
        index_path: str,
        embedding_dim: int,
        metric: str = "cos",
        dtype: str = "f16",
        connectivity: int = 32,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ):
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.dtype = dtype
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        
        self.index = self._init_index()
    
    def _init_index(self, view_only: bool = False) -> USearchIndex:
        """Initialize or load the USearch index."""
        if self.index_path.exists():
            index = USearchIndex.restore(str(self.index_path), view=view_only)
        else:
            # Create new index
            metric_kind = {
                "cos": MetricKind.Cosine,
                "ip": MetricKind.IP,
                "l2sq": MetricKind.L2sq,
            }.get(self.metric, MetricKind.Cosine)
            
            index = USearchIndex(
                ndim=self.embedding_dim,
                metric=metric_kind,
                dtype=self.dtype,
                connectivity=self.connectivity,
                expansion_add=self.expansion_add,
                expansion_search=self.expansion_search,
            )
        
        return index
    
    def add(self, key: int, embedding: List[float]) -> None:
        """Add embedding to index."""
        self.index.add(key, np.array(embedding, dtype=np.float32))
    
    def search(self, embedding: List[float], k: int = 10) -> Matches:
        """Search for nearest neighbors."""
        return self.index.search(np.array(embedding, dtype=np.float32), k)
    
    def remove(self, key: int) -> None:
        """Remove embedding from index."""
        self.index.remove(key)
    
    def save(self) -> None:
        """Persist index to disk."""
        self.index.save(str(self.index_path))
    
    def __len__(self) -> int:
        """Get number of vectors in index."""
        return len(self.index)
