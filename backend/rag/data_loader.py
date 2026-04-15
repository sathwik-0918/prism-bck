# rag/data_loader.py
# Added OCR fallback for scanned/image-based PDFs
# Uses Tesseract OCR via pytesseract when both PyMuPDF and pdfplumber fail

import os
os.environ["PYMUPDF_MAX_STORE"] = str(500 * 1024 * 1024)

import fitz
import pdfplumber
import unicodedata
import re
from pathlib import Path
from typing import List, Any, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader

# OCR imports — graceful fallback if not installed
try:
    import pytesseract
    from PIL import Image
    try:
        from pdf2image import convert_from_path
        PDF2IMAGE_AVAILABLE = True
    except ImportError:
        convert_from_path = None
        PDF2IMAGE_AVAILABLE = False

    # Windows — set tesseract path if not in PATH
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    OCR_AVAILABLE = True
    print("[INFO] OCR available — Tesseract loaded for scanned PDFs")
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] OCR not available — install pytesseract + pdf2image for scanned PDF support")


def clean_text(text: str) -> str:
    """Cleans extracted text preserving all math symbols."""
    if not text:
        return ""

    # remove ONLY surrogates (not Greek letters, math symbols etc)
    cleaned = []
    for char in text:
        code = ord(char)
        if not (0xD800 <= code <= 0xDFFF):
            cleaned.append(char)
    text = ''.join(cleaned)

    # normalize unicode (NFC preserves all symbols)
    try:
        text = unicodedata.normalize("NFC", text)
    except Exception:
        pass

    # remove only true control chars (keep newlines, tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return text


def ocr_page(pdf_path: str, page_num: int) -> str:
    """
    Uses Tesseract OCR to extract text from a scanned PDF page.
    Converts page to high-resolution image first, then runs OCR.
    """
    if not OCR_AVAILABLE:
        return ""

    try:
        img = None

        # First choice: render directly with PyMuPDF so OCR does not depend on Poppler.
        try:
            pdf = fitz.open(pdf_path)
            page = pdf[page_num]
            matrix = fitz.Matrix(2.5, 2.5)  # ~250 DPI equivalent
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf.close()
        except Exception:
            img = None

        # Second choice: pdf2image if available.
        if img is None and PDF2IMAGE_AVAILABLE:
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300,
                fmt="PNG"
            )
            if images:
                img = images[0]

        if img is None:
            return ""

        # OCR config — PSM 3 = fully automatic page segmentation
        # OEM 3 = default LSTM OCR engine (most accurate)
        custom_config = r'--oem 3 --psm 3'
        text = pytesseract.image_to_string(img, config=custom_config, lang='eng')

        return clean_text(text) if text else ""

    except Exception as e:
        print(f"[OCR] Failed on page {page_num}: {e}")
        return ""


def load_pdf_with_fallback(pdf_path: str) -> List[Document]:
    """
    3-tier PDF loading:
    1. PyMuPDF (fastest, handles most PDFs)
    2. pdfplumber (handles SymbolSetEncoding)
    3. Tesseract OCR (handles scanned/image PDFs)
    """
    documents = []
    path = str(pdf_path)
    filename = Path(path).name

    try:
        pdf = fitz.open(path)
        total_pages = pdf.page_count
        ocr_pages = 0
        loaded_pages = 0

        print(f"[INFO] Loading PDF ({total_pages} pages): {filename}")

        for page_num in range(total_pages):
            page = pdf[page_num]
            page_doc = None

            # ── TIER 1: PyMuPDF ──────────────────────────────────
            try:
                raw_text = page.get_text("text")
                cleaned = clean_text(raw_text)
                if len(cleaned.strip()) >= 20:
                    page_doc = Document(
                        page_content=cleaned,
                        metadata={"source": filename, "page": page_num + 1, "loader": "pymupdf"}
                    )
            except Exception:
                pass

            # ── TIER 2: pdfplumber ────────────────────────────────
            if not page_doc:
                try:
                    with pdfplumber.open(path) as plumber_pdf:
                        if page_num < len(plumber_pdf.pages):
                            raw = plumber_pdf.pages[page_num].extract_text() or ""

                            # also extract tables as text
                            tables = plumber_pdf.pages[page_num].extract_tables()
                            table_text = ""
                            for table in (tables or []):
                                for row in table:
                                    row_text = " | ".join([str(cell or "").strip() for cell in row if cell])
                                    if row_text.strip():
                                        table_text += row_text + "\n"

                            combined = clean_text(raw + "\n" + table_text)
                            if len(combined.strip()) >= 20:
                                page_doc = Document(
                                    page_content=combined,
                                    metadata={"source": filename, "page": page_num + 1, "loader": "pdfplumber"}
                                )
                except Exception:
                    pass

            # ── TIER 3: OCR (for scanned/image PDFs) ─────────────
            if not page_doc and OCR_AVAILABLE:
                ocr_text = ocr_page(path, page_num)
                if len(ocr_text.strip()) >= 20:
                    page_doc = Document(
                        page_content=ocr_text,
                        metadata={"source": filename, "page": page_num + 1, "loader": "ocr_tesseract"}
                    )
                    ocr_pages += 1
                    print(f"[OCR] Page {page_num+1} recovered via Tesseract: {filename}")

            if page_doc and len(page_doc.page_content.strip()) >= 20:
                documents.append(page_doc)
                loaded_pages += 1

        pdf.close()
        print(f"[INFO] {filename}: {loaded_pages}/{total_pages} pages loaded "
              f"({ocr_pages} via OCR)")

    except Exception as e:
        print(f"[ERROR] Cannot open PDF {filename}: {e}")

    return documents


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Loads all documents with 3-tier extraction.
    Handles: normal PDFs, SymbolEncoding PDFs, scanned/image PDFs.
    """
    data_path = Path(data_dir).resolve()
    print(f"\n[DEBUG] Scanning: {data_path}")
    documents = []

    # PDFs
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        try:
            loaded = load_pdf_with_fallback(str(pdf_file))
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] PDF failed {pdf_file.name}: {e}")

    # TXT files
    txt_files = list(data_path.glob("**/*.txt"))
    for txt_file in txt_files:
        try:
            for encoding in ["utf-8", "latin-1"]:
                try:
                    loader = TextLoader(str(txt_file), encoding=encoding)
                    loaded = loader.load()
                    for doc in loaded:
                        doc.page_content = clean_text(doc.page_content)
                        doc.metadata["source"] = txt_file.name
                    documents.extend(loaded)
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"[ERROR] TXT {txt_file.name}: {e}")

    # Word files
    word_files = list(data_path.glob("**/*.docx"))
    for word_file in word_files:
        try:
            loader = Docx2txtLoader(str(word_file))
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = word_file.name
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Word {word_file.name}: {e}")

    print(f"\n[INFO] Total documents loaded: {len(documents)}")
    return documents
