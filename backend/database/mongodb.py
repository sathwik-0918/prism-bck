# database/mongodb.py
# MongoDB connection manager
# motor is async MongoDB driver — perfect for FastAPI
# think of this like mongoose.connect() in your blog app

from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGODB_URI, MONGODB_DB
import asyncio

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

mongodb = MongoDB()

async def connect_db():
    """
    Connect to MongoDB on server startup.
    Like mongoose.connect() in Express.
    """
    print("[INFO] Connecting to MongoDB...")
    mongodb.client = AsyncIOMotorClient(MONGODB_URI)
    mongodb.db = mongodb.client[MONGODB_DB]
    print(f"[INFO] MongoDB connected — database: {MONGODB_DB}")

async def close_db():
    """
    Close MongoDB connection on server shutdown.
    """
    if mongodb.client:
        mongodb.client.close()
        print("[INFO] MongoDB connection closed.")

def get_db():
    """
    Returns database instance.
    Used in all API files to get DB access.
    Like importing mongoose models in Express.
    """
    return mongodb.db