# -*- coding: utf-8 -*-
# rag/data_loader.py
# loads all documents from data folder into LangChain Document structure
# uses PyMuPDF as primary PDF loader
# falls back to pdfplumber for SymbolSetEncoding and encoding issues
# handles decompression limits for large PDFs (Arihant handbooks 500+ pages)
# cleans all text at load time — surrogates, control chars, null bytes removed
# skips users.json — not academic content

# increase PyMuPDF decompression limit BEFORE importing fitz
# default is ~20MB — too small for Arihant handbooks
# setting to 500MB handles all large academic PDFs
import os
os.environ["PYMUPDF_MAX_STORE"] = str(500 * 1024 * 1024)

import fitz  # PyMuPDF — import after env var is set
import pdfplumber
import unicodedata
import re
from pathlib import Path
from typing import List, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader


# ─────────────────────────────────────────────
# TEXT CLEANER
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Professional-grade text cleaner for academic PDF content.
    Handles all known issues from JEE/NEET PDF extraction:

    1. Surrogate characters (e.g., \\ud835) — crash utf-8 encoding
    2. Null bytes — corrupt embeddings
    3. Non-printable control characters — noise in embeddings
    4. Excessive whitespace — wastes token space
    5. Unicode normalization — standardizes math symbols (∫ α β θ ∑)

    This runs at LOAD TIME so every document entering the pipeline is clean.
    embedding.py also cleans at CHUNK TIME as a second safety layer.
    """
    if not text:
        return ""

    # step 1 — remove surrogate characters (\\ud800-\\udfff)
    # these are the main cause of 'utf-8 codec cant encode' crashes
    # surrogatepass encodes them, ignore drops them cleanly
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")

    # step 2 — unicode normalization
    # NFKC standardizes math symbols — converts special variants to standard
    # example: ℕ → N, ² → 2, ﬁ → fi, α stays α, ∫ stays ∫
    text = unicodedata.normalize("NFKC", text)

    # step 3 — remove null bytes and non-printable control characters
    # keep \n (newline=0x0a) and \t (tab=0x09) — needed for structure
    # remove everything else in 0x00-0x1f range and 0x7f (DEL)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # step 4 — remove leftover surrogate escape sequences in string form
    # sometimes PDF text has literal \ud835 as text — remove those too
    text = re.sub(r"\\ud[0-9a-f]{3}", "", text, flags=re.IGNORECASE)

    # step 5 — normalize excessive whitespace while keeping document structure
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)   # max 1 space or tab
    text = text.strip()

    # step 6 — final safety encode check
    # if anything still cant encode after all above — strip it
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return text


# ─────────────────────────────────────────────
# PDF LOADER WITH FALLBACK
# ─────────────────────────────────────────────

def load_pdf_with_fallback(pdf_path: str) -> List[Document]:
    """
    Loads a PDF page by page using PyMuPDF as primary loader.
    Falls back to pdfplumber for pages with:
      - SymbolSetEncoding (math symbols in old PDFs)
      - Insufficient text extraction (< 20 chars)
      - Any PyMuPDF page-level exception

    This dual-loader approach ensures maximum content recovery
    from all JEE/NEET PDFs including old question papers (2013-2019)
    and Arihant handbooks with special symbol encodings.

    Returns list of Document objects — one per page — with metadata:
      - source: filename
      - page: page number (1-indexed)
      - loader: 'pymupdf' or 'pdfplumber_fallback'
    """
    documents = []
    path = str(pdf_path)
    filename = Path(path).name

    try:
        pdf = fitz.open(path)
        total_pages = pdf.page_count
        recovered_pages = 0
        failed_pages = 0

        print(f"[INFO] Loading PDF ({total_pages} pages): {filename}")

        for page_num in range(total_pages):
            page = pdf[page_num]
            page_doc = None

            # ── PRIMARY: PyMuPDF ──────────────────────────
            try:
                raw_text = page.get_text("text")
                cleaned = clean_text(raw_text)

                if len(cleaned.strip()) >= 20:
                    # good extraction — use PyMuPDF result
                    page_doc = Document(
                        page_content=cleaned,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "loader": "pymupdf"
                        }
                    )
                else:
                    # text too short — trigger fallback
                    raise ValueError(f"Insufficient text ({len(cleaned.strip())} chars)")

            except Exception as pymupdf_err:
                # ── FALLBACK: pdfplumber ──────────────────
                try:
                    with pdfplumber.open(path) as plumber_pdf:
                        if page_num < len(plumber_pdf.pages):
                            plumber_page = plumber_pdf.pages[page_num]

                            # extract text — pdfplumber handles SymbolSetEncoding
                            raw_text = plumber_page.extract_text() or ""
                            cleaned = clean_text(raw_text)

                            if cleaned.strip():
                                page_doc = Document(
                                    page_content=cleaned,
                                    metadata={
                                        "source": filename,
                                        "page": page_num + 1,
                                        "loader": "pdfplumber_fallback"
                                    }
                                )
                                recovered_pages += 1
                                print(f"[INFO] Page {page_num+1} recovered via pdfplumber: {filename}")
                            else:
                                # both loaders got empty text — blank/image page
                                failed_pages += 1
                                print(f"[WARN] Page {page_num+1} empty after both loaders: {filename}")

                except Exception as plumber_err:
                    failed_pages += 1
                    print(f"[WARN] Page {page_num+1} failed both loaders.")
                    print(f"       PyMuPDF: {pymupdf_err}")
                    print(f"       pdfplumber: {plumber_err}")

            # add to documents if we got content
            if page_doc and len(page_doc.page_content.strip()) >= 20:
                documents.append(page_doc)

        pdf.close()

        print(f"[INFO] {filename}: {len(documents)}/{total_pages} pages loaded "
              f"({recovered_pages} via fallback, {failed_pages} failed)")

    except Exception as e:
        print(f"[ERROR] Cannot open PDF {filename}: {e}")

    return documents


# ─────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Loads all supported academic files from data directory recursively.

    Supported formats:
      - PDF  → PyMuPDF + pdfplumber fallback (handles all JEE/NEET PDFs)
      - TXT  → TextLoader with utf-8
      - CSV  → CSVLoader
      - DOCX → Docx2txtLoader

    Explicitly skips:
      - users.json — user data, not academic content
      - Any other JSON — not relevant to RAG knowledge base

    All text is cleaned at load time via clean_text().
    Additional cleaning happens at chunk time in embedding.py.
    Two-layer cleaning = zero encoding crashes in the pipeline.
    """
    data_path = Path(data_dir).resolve()
    print(f"\n[DEBUG] Scanning data directory: {data_path}")
    documents = []

    # ── PDF FILES ──────────────────────────────────────────────────────
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        try:
            loaded = load_pdf_with_fallback(str(pdf_file))
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Unexpected error loading PDF {pdf_file.name}: {e}")

    # ── TXT FILES ──────────────────────────────────────────────────────
    txt_files = list(data_path.glob("**/*.txt"))
    print(f"\n[DEBUG] Found {len(txt_files)} TXT files")

    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()

            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = txt_file.name
                doc.metadata["loader"] = "textloader"

            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} docs from {txt_file.name}")

        except UnicodeDecodeError:
            # retry with latin-1 for files with non-utf8 encoding
            try:
                loader = TextLoader(str(txt_file), encoding="latin-1")
                loaded = loader.load()
                for doc in loaded:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata["source"] = txt_file.name
                    doc.metadata["loader"] = "textloader_latin1"
                documents.extend(loaded)
                print(f"[DEBUG] Loaded {txt_file.name} with latin-1 fallback")
            except Exception as e:
                print(f"[ERROR] TXT failed both encodings {txt_file.name}: {e}")

        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file.name}: {e}")

    # ── CSV FILES ──────────────────────────────────────────────────────
    csv_files = list(data_path.glob("**/*.csv"))
    print(f"\n[DEBUG] Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = csv_file.name
            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} docs from {csv_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file.name}: {e}")

    # ── WORD FILES ─────────────────────────────────────────────────────
    word_files = list(data_path.glob("**/*.docx"))
    print(f"\n[DEBUG] Found {len(word_files)} Word files")

    for word_file in word_files:
        try:
            loader = Docx2txtLoader(str(word_file))
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = word_file.name
                doc.metadata["loader"] = "docx2txt"
            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} docs from {word_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load Word {word_file.name}: {e}")

    # ── SUMMARY ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"[INFO] Total documents loaded: {len(documents)}")
    print(f"[INFO] Sources: PDFs={len(pdf_files)}, "
          f"TXT={len(txt_files)}, "
          f"CSV={len(csv_files)}, "
          f"Word={len(word_files)}")
    print(f"{'='*50}\n")

    return documents