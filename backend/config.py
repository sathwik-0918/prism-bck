# config.py
from dotenv import load_dotenv
import os

load_dotenv()

PORT = int(os.getenv("PORT", 8000))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
FAISS_STORE_PATH = os.getenv("FAISS_STORE_PATH", "./faiss_store")
DATA_PATH = os.getenv("DATA_PATH", "./data")