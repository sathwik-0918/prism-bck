# database/mongodb.py
# MongoDB connection manager
# motor is async MongoDB driver — perfect for FastAPI
# think of this like mongoose.connect() in your blog app

from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGODB_URI, MONGODB_DB
import asyncio
import os
from dotenv import load_dotenv


load_dotenv()

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None
    atlas_client: AsyncIOMotorClient = None
    atlas_db = None

mongodb = MongoDB()

ATLAS_URI = os.getenv("MONGODB_ATLAS_URI", "")
ATLAS_DB = os.getenv("MONGODB_ATLAS_DB", "prism_chat")

async def connect_db():
    """
    Connect to MongoDB on server startup.
    Like mongoose.connect() in Express.
    """
    print("[INFO] Connecting to MongoDB...")
    mongodb.client = AsyncIOMotorClient(MONGODB_URI)
    mongodb.db = mongodb.client[MONGODB_DB]
    print(f"[INFO] MongoDB connected — database: {MONGODB_DB}")

    # connect Atlas for chat if configured
    if ATLAS_URI:
        print("[INFO] Connecting to MongoDB Atlas for chat...")
        mongodb.atlas_client = AsyncIOMotorClient(ATLAS_URI)
        mongodb.atlas_db = mongodb.atlas_client[ATLAS_DB]
        print("[INFO] Atlas connected — chat data will sync across devices")
    else:
        print("[INFO] No Atlas URI — chat uses local DB (single device only)")

async def close_db():
    """
    Close MongoDB connection on server shutdown.
    """
    if mongodb.client:
        mongodb.client.close()
        print("[INFO] MongoDB connection closed.")
    if mongodb.atlas_client:
        mongodb.atlas_client.close()
    print("[INFO] MongoDB connections closed.")

def get_db():
    """
    Returns database instance.
    Used in all API files to get DB access.
    Like importing mongoose models in Express.
    """
    return mongodb.db

def get_chat_db():
    """
    Returns Atlas DB for chat if configured, otherwise local.
    """
    if mongodb.atlas_db is not None:
        return mongodb.atlas_db
    return mongodb.db  # fallback to local