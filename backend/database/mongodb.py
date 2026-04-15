# database/mongodb.py
# MongoDB connection manager
# motor is async MongoDB driver — perfect for FastAPI
# think of this like mongoose.connect() in your blog app
# database/mongodb.py
# Local DB → all features (RAG, quiz, planner, etc.)
# Atlas DB → leaderboard + chat users only (cross-device)


from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGODB_URI, MONGODB_DB
import os
from config import MONGODB_ATLAS_URI, MONGODB_ATLAS_DB

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None
    atlas_client: AsyncIOMotorClient = None
    atlas_db = None

mongodb = MongoDB()

ATLAS_URI = os.getenv("MONGODB_ATLAS_URI", MONGODB_ATLAS_URI)
ATLAS_DB_NAME = os.getenv("MONGODB_ATLAS_DB", MONGODB_ATLAS_DB)

async def connect_db():
    print("[INFO] Connecting to local MongoDB...")
    mongodb.client = AsyncIOMotorClient(MONGODB_URI)
    mongodb.db = mongodb.client[MONGODB_DB]
    print(f"[INFO] Local MongoDB ready — {MONGODB_DB}")

    if ATLAS_URI:
        print("[INFO] Connecting to MongoDB Atlas (cloud)...")
        mongodb.atlas_client = AsyncIOMotorClient(ATLAS_URI)
        mongodb.atlas_db = mongodb.atlas_client[ATLAS_DB_NAME]
        print(f"[INFO] Atlas ready — {ATLAS_DB_NAME} (leaderboard + chat users)")
    else:
        print("[WARN] No Atlas URI — leaderboard is local only")

async def close_db():
    if mongodb.client:
        mongodb.client.close()
    if mongodb.atlas_client:
        mongodb.atlas_client.close()
    print("[INFO] DB connections closed.")

def get_db():
    """
    Returns database instance.
    Used in all API files to get DB access.
    Like importing mongoose models in Express.
    """
    return mongodb.db

def get_cloud_db():
    """
    Atlas DB for cross-device features.
    Falls back to local if Atlas not configured.
    Used by: leaderboard, studychat users, friend system
    """
    if mongodb.atlas_db is not None:
        return mongodb.atlas_db
    return mongodb.db


def get_chat_db():
    return get_cloud_db()
