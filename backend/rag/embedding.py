# -*- coding: utf-8 -*-
# rag/embedding.py
# handles document chunking and embedding generation
# uses RecursiveCharacterTextSplitter for semantic chunking
# uses all-MiniLM-L6-v2 sentence transformer for embeddings
# deep cleans every chunk before embedding — zero encoding crashes

from typing import List, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import unicodedata
import re


# ─────────────────────────────────────────────
# TEXT CLEANER
# ─────────────────────────────────────────────

def deep_clean_text(text: str) -> str:
    """
    Professional-grade text cleaner for academic PDF content.
    Handles all known issues from JEE/NEET PDF extraction:

    1. Surrogate characters — crash utf-8 encoding
    2. Null bytes — corrupt embeddings
    3. Non-printable control characters — noise in embeddings
    4. Excessive whitespace — wastes token space
    5. Unicode normalization — standardizes math symbols
    """
    if not text:
        return ""

    # step 1 — remove surrogate characters
    # encode to utf-16 with surrogatepass then decode back ignoring them
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    except Exception:
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # step 2 — unicode normalization
    # NFKC standardizes math symbols — keeps alpha, beta, integral etc intact
    try:
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        pass

    # step 3 — remove null bytes and non-printable control characters
    # keep newline (0x0a) and tab (0x09) — needed for document structure
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # step 4 — remove surrogate escape sequences in string form
    text = re.sub(r"\\ud[0-9a-fA-F]{3}", "", text)

    # step 5 — normalize excessive whitespace while keeping structure
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    # step 6 — final safety encode check
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return text


# ─────────────────────────────────────────────
# EMBEDDING PIPELINE
# ─────────────────────────────────────────────

class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        """
        Initializes the embedding pipeline.

        chunk_size=800 — better than 1000 for JEE/NEET content.
        Keeps one concept per chunk for precise retrieval.

        chunk_overlap=150 — ensures formulas spanning chunk
        boundaries are not lost between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")
        print(f"[INFO] Chunk size: {chunk_size} | Overlap: {chunk_overlap}")

    def chunk_documents(self, documents: List[Any]) -> List[Document]:
        """
        Splits documents into semantic chunks.

        Separators ordered from largest to smallest unit.
        Tries paragraph breaks first, then sentences, then words.
        This keeps physics derivations and chemistry equations intact.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",   # paragraph break — best split point
                "\n",     # line break
                ". ",     # sentence end
                "? ",     # question end — good for PYQs
                "! ",     # exclamation
                "; ",     # semicolon
                ", ",     # comma
                " ",      # word
                ""        # character — last resort
            ]
        )

        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def clean_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Deep cleans every chunk before embedding.
        Removes surrogates, control chars, normalizes unicode.

        Skips chunks that are empty or too short after cleaning.
        Under 30 chars = likely just a page number or blank page.
        Not worth embedding — wastes vector space.
        """
        cleaned = []
        skipped = 0

        for chunk in chunks:
            cleaned_text = deep_clean_text(chunk.page_content)

            # skip near-empty chunks
            if len(cleaned_text.strip()) < 30:
                skipped += 1
                continue

            cleaned.append(Document(
                page_content=cleaned_text,
                metadata=chunk.metadata
            ))

        print(f"[INFO] Chunks after cleaning: {len(cleaned)} kept, {skipped} skipped.")
        return cleaned

    def embed_chunks(self, chunks: List[Document]) -> Tuple[np.ndarray, List[Document]]:
        """
        Generates embeddings for all cleaned chunks.

        Returns (embeddings, cleaned_chunks) as a tuple.
        vectorstore.py must unpack both:
            embeddings, cleaned_chunks = emb_pipe.embed_chunks(chunks)

        normalize_embeddings=True — normalizes vectors for better
        cosine similarity search. Must match query embedding normalization.

        batch_size=64 — processes 64 chunks at a time for memory efficiency.
        """
        # clean all chunks — zero surrogates reach the encoder
        cleaned_chunks = self.clean_chunks(chunks)

        texts = [chunk.page_content for chunk in cleaned_chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True
        )

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings, cleaned_chunks