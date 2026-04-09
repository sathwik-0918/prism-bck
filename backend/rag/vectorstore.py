# rag/vectorstore.py
# handles FAISS vector store operations
# build, save, load, search — complete vector database management
# stores embeddings + rich metadata (source, page, loader) for accurate retrieval

import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from rag.embedding import EmbeddingPipeline, deep_clean_text


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        """
        Initializes the FAISS vector store.
        persist_dir — where faiss_index and metadata.pkl are saved.
        chunk_size=800 keeps one concept per chunk for JEE/NEET content.
        chunk_overlap=150 prevents formulas from being split across chunks.
        """
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # separate model instance for query-time embedding
        self.model = SentenceTransformer(self.embedding_model)
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        """
        Full pipeline: raw docs → chunk → deep clean → embed → store.
        No data is compromised — surrogates cleaned, symbols recovered,
        empty pages skipped, everything else kept with full metadata.
        """
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # step 1 — chunk documents
        chunks = emb_pipe.chunk_documents(documents)

        # step 2 — deep clean + embed
        # returns (embeddings, cleaned_chunks) — no surrogates, no garbage
        embeddings, cleaned_chunks = emb_pipe.embed_chunks(chunks)

        # step 3 — build rich metadata for every chunk
        # source + page + loader tracked for citation display in frontend
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "Unknown Source"),
                "page": chunk.metadata.get("page", 0),
                "loader": chunk.metadata.get("loader", "unknown")
            }
            for chunk in cleaned_chunks
        ]

        # step 4 — add to FAISS index
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)

        # step 5 — persist to disk
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")
        print(f"[INFO] Total vectors stored: {self.index.ntotal}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        """
        Adds embedding vectors to FAISS index.
        Uses IndexFlatL2 — exact L2 distance search.
        For 10k+ vectors this is accurate and fast enough for our use case.
        """
        dim = embeddings.shape[1]

        if self.index is None:
            # initialize index on first call
            self.index = faiss.IndexFlatL2(dim)
            print(f"[INFO] FAISS index initialized with dimension: {dim}")

        self.index.add(embeddings)

        if metadatas is not None:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors. Total: {self.index.ntotal}")

    def save(self):
        """
        Saves FAISS index and metadata to disk.
        faiss_index — binary index file (fast load)
        metadata.pkl — python list of dicts (source, page, text, loader)
        """
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index ({self.index.ntotal} vectors) to {self.persist_dir}")
        print(f"[INFO] Saved {len(self.metadata)} metadata entries.")

    def load(self):
        """
        Loads existing FAISS index and metadata from disk.
        Called on server startup if index already exists —
        skips the 5 minute rebuild process.
        """
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded FAISS index: {self.index.ntotal} vectors.")
        print(f"[INFO] Loaded {len(self.metadata)} metadata entries.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Searches FAISS index for top_k nearest vectors.
        Returns list of dicts with index, distance, and full metadata.
        Lower distance = more similar to query.
        """
        if self.index is None or self.index.ntotal == 0:
            print("[WARN] FAISS index is empty — no results returned.")
            return []

        # cap top_k to available vectors
        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, top_k)
        results = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                # invalid index — skip
                continue

            meta = self.metadata[idx]

            # skip chunks with no meaningful text
            if not meta.get("text", "").strip():
                continue

            results.append({
                "index": int(idx),
                "distance": float(dist),
                "metadata": meta
            })

        print(f"[INFO] Search returned {len(results)} results.")
        return results

    def query(self, query_text: str, top_k: int = 5):
        """
        End-to-end query: text → embed → search → return results.
        Cleans query text before embedding for consistency.
        Called by nodes.py retrieve_node for every user question.
        """
        print(f"[INFO] Querying vector store: '{query_text[:80]}...'")

        # clean query same way we cleaned chunks — consistent vector space
        cleaned_query = deep_clean_text(query_text)

        # embed query with normalization matching how chunks were embedded
        query_embedding = self.model.encode(
            [cleaned_query],
            normalize_embeddings=True   # must match embed_chunks normalization
        ).astype("float32")

        return self.search(query_embedding, top_k)