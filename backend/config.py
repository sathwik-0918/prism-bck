from dotenv import load_dotenv
import os

load_dotenv()

PORT = int(os.getenv("PORT", 8000))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./embeddings")