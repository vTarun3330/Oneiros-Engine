"""
FAISS Memory Module for Oneiros Engine.

This module implements semantic memory using FAISS for storing and retrieving
successful test inputs. It enables novelty checking and retrieval of similar
past inputs for prompting Phi-3.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss-cpu not installed. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import memory_config, DATA_DIR


@dataclass
class MemoryEntry:
    """Represents an entry in the memory."""
    id: str
    test_input: str              # The test input that was successful
    function_id: str             # Which function this was for
    embedding: List[float]       # Embedding vector (stored for serialization)
    found_bug: bool              # Whether this input found a bug
    is_novel: bool               # Whether this was novel when added
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FAISSMemory:
    """
    Semantic memory using FAISS for test input storage and retrieval.

    This is CRITICAL for system-level testing where each test takes 200ms+.
    We filter redundant tests to avoid wasting execution time.
    """

    def __init__(
        self,
        embedding_dim: int = None,
        embedding_model: str = None,
        novelty_threshold: float = None
    ):
        """
        Initialize FAISS memory.

        Args:
            embedding_dim: Dimension of embeddings (default from config)
            embedding_model: SentenceTransformer model name
            novelty_threshold: Cosine similarity threshold for novelty
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        self.embedding_dim = embedding_dim or memory_config.embedding_dim
        self.model_name = embedding_model or memory_config.embedding_model
        self.novelty_threshold = novelty_threshold or memory_config.novelty_threshold
        self.max_size = memory_config.max_memory_size

        # Initialize embedding model
        self.encoder = SentenceTransformer(self.model_name)

        # Initialize FAISS index (using Inner Product for cosine similarity)
        # We normalize vectors, so IP = cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Store metadata alongside embeddings
        self.entries: List[MemoryEntry] = []
        self.id_to_idx: Dict[str, int] = {}

        # Statistics
        self.stats = {
            "total_added": 0,
            "bugs_found": 0,
            "novel_inputs": 0,
            "duplicates_filtered": 0
        }

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.

        Args:
            text: Input text to encode

        Returns:
            Normalized embedding vector
        """
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts to embeddings."""
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        # Normalize each vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings

    def is_novel(self, text: str) -> Tuple[bool, float]:
        """
        Check if a test input is novel (different from existing memory).

        Args:
            text: The test input to check

        Returns:
            Tuple of (is_novel, max_similarity)
        """
        if self.index.ntotal == 0:
            return True, 0.0

        embedding = self.encode(text)
        embedding = embedding.reshape(1, -1)

        # Search for most similar
        distances, indices = self.index.search(embedding, k=1)
        max_similarity = float(distances[0][0])

        # Novel if similarity is below threshold
        is_novel = max_similarity < self.novelty_threshold

        if not is_novel:
            self.stats["duplicates_filtered"] += 1

        return is_novel, max_similarity

    def add(
        self,
        test_input: str,
        function_id: str,
        found_bug: bool = False,
        metadata: Dict[str, Any] = None,
        check_novelty: bool = True
    ) -> Optional[MemoryEntry]:
        """
        Add a test input to memory.

        Args:
            test_input: The test input string
            function_id: ID of the function this was for
            found_bug: Whether this input found a bug
            metadata: Additional metadata
            check_novelty: Whether to check novelty before adding

        Returns:
            MemoryEntry if added, None if duplicate
        """
        # Check novelty if requested
        is_novel = True
        if check_novelty:
            is_novel, similarity = self.is_novel(test_input)
            if not is_novel:
                return None

        # Encode and add
        embedding = self.encode(test_input)

        entry_id = f"mem_{len(self.entries)}"
        entry = MemoryEntry(
            id=entry_id,
            test_input=test_input,
            function_id=function_id,
            embedding=embedding.tolist(),
            found_bug=found_bug,
            is_novel=is_novel,
            metadata=metadata or {}
        )

        # Add to FAISS
        self.index.add(embedding.reshape(1, -1))

        # Add to metadata storage
        self.entries.append(entry)
        self.id_to_idx[entry_id] = len(self.entries) - 1

        # Update stats
        self.stats["total_added"] += 1
        if found_bug:
            self.stats["bugs_found"] += 1
        if is_novel:
            self.stats["novel_inputs"] += 1

        return entry

    def add_batch(
        self,
        test_inputs: List[str],
        function_id: str,
        found_bugs: List[bool] = None
    ) -> List[MemoryEntry]:
        """Add multiple test inputs at once."""
        found_bugs = found_bugs or [False] * len(test_inputs)
        entries = []

        for text, found_bug in zip(test_inputs, found_bugs):
            entry = self.add(
                test_input=text,
                function_id=function_id,
                found_bug=found_bug
            )
            if entry:
                entries.append(entry)

        return entries

    def retrieve_similar(
        self,
        query: str,
        k: int = 5,
        function_id: str = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve similar test inputs from memory.

        Args:
            query: Query text
            k: Number of results to return
            function_id: Optional filter by function

        Returns:
            List of (MemoryEntry, similarity) tuples
        """
        if self.index.ntotal == 0:
            return []

        embedding = self.encode(query)
        embedding = embedding.reshape(1, -1)

        # Search more than k to allow for filtering
        search_k = min(k * 2, self.index.ntotal)
        distances, indices = self.index.search(embedding, k=search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.entries):
                continue

            entry = self.entries[idx]

            # Filter by function if specified
            if function_id and entry.function_id != function_id:
                continue

            results.append((entry, float(dist)))

            if len(results) >= k:
                break

        return results

    def get_winners(self, k: int = None) -> List[MemoryEntry]:
        """Get test inputs that found bugs (winners for DPO)."""
        winners = [e for e in self.entries if e.found_bug]
        if k:
            winners = winners[:k]
        return winners

    def get_for_prompt(
        self,
        function_id: str,
        k: int = 3
    ) -> List[str]:
        """
        Get example inputs to include in LLM prompt.

        Args:
            function_id: Function to get examples for
            k: Number of examples

        Returns:
            List of test input strings
        """
        # Prioritize inputs that found bugs
        winners = [e for e in self.entries
                   if e.function_id == function_id and e.found_bug][:k]

        # Fill remaining with novel inputs
        if len(winners) < k:
            others = [e for e in self.entries
                      if e.function_id == function_id
                      and not e.found_bug
                      and e.is_novel][:k - len(winners)]
            winners.extend(others)

        return [e.test_input for e in winners]

    def save(self, path: Path = None) -> Path:
        """Save memory to disk."""
        path = path or (DATA_DIR / "faiss_memory")
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata
        metadata = {
            "entries": [e.to_dict() for e in self.entries],
            "id_to_idx": self.id_to_idx,
            "stats": self.stats,
            "config": {
                "embedding_dim": self.embedding_dim,
                "model_name": self.model_name,
                "novelty_threshold": self.novelty_threshold
            }
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved memory with {len(self.entries)} entries to {path}")
        return path

    def load(self, path: Path = None) -> None:
        """Load memory from disk."""
        path = path or (DATA_DIR / "faiss_memory")

        if not (path / "index.faiss").exists():
            raise FileNotFoundError(f"Memory not found at {path}")

        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))

        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        self.entries = [MemoryEntry(**e) for e in metadata["entries"]]
        self.id_to_idx = metadata["id_to_idx"]
        self.stats = metadata["stats"]

        print(f"Loaded memory with {len(self.entries)} entries from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self.stats,
            "current_size": len(self.entries),
            "index_size": self.index.ntotal
        }

    def clear(self) -> None:
        """Clear all memory."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.entries = []
        self.id_to_idx = {}
        self.stats = {
            "total_added": 0,
            "bugs_found": 0,
            "novel_inputs": 0,
            "duplicates_filtered": 0
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FAISS Memory")
    print("=" * 60)

    memory = FAISSMemory()

    print("\n1. Adding test inputs...")
    memory.add("result = merge_wrapper({'a': [1,2]}, {'b': [3,4]}, on='a')", "sys_pandas_merge", found_bug=True)
    memory.add("result = merge_wrapper({}, {}, on='key')", "sys_pandas_merge", found_bug=True)
    memory.add("result = json_loads_wrapper('{\"key\": \"value\"}')", "sys_json_loads")
    memory.add("result = json_loads_wrapper('invalid json')", "sys_json_loads", found_bug=True)

    print("\n2. Checking novelty...")
    is_novel, sim = memory.is_novel("result = merge_wrapper({'a': [1,2]}, {'b': [3,4]}, on='a')")
    print(f"   Same input novel? {is_novel} (similarity: {sim:.3f})")

    is_novel, sim = memory.is_novel("result = merge_wrapper({'x': [9,9]}, {'y': [8,8]}, on='x')")
    print(f"   Similar input novel? {is_novel} (similarity: {sim:.3f})")

    print("\n3. Retrieving similar inputs...")
    similar = memory.retrieve_similar("merge two dataframes", k=3)
    for entry, score in similar:
        print(f"   {entry.test_input[:50]}... (score: {score:.3f})")

    print("\n4. Getting winners for DPO...")
    winners = memory.get_winners()
    print(f"   Found {len(winners)} winners")

    print("\n5. Stats:")
    stats = memory.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("FAISS Memory test complete!")
    print("=" * 60)
