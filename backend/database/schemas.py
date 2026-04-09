# database/schemas.py
# Pydantic schemas for MongoDB documents
# like Mongoose schemas in your blog app
# Message → inside Session → belongs to User

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class MessageSchema(BaseModel):
    """
    Single message in a chat session.
    role: 'user' or 'assistant'
    sources: list of PDF source citations
    timestamp: when message was sent
    """
    role: str
    content: str
    sources: List[str] = []
    timestamp: str = ""


class ChatSessionSchema(BaseModel):
    """
    A full chat session — one conversation thread.
    Like an article in your blog app.
    Contains multiple messages.
    """
    sessionId: str
    userId: str
    title: str                      # auto-generated from first message
    examTarget: str                 # JEE or NEET
    messages: List[MessageSchema] = []
    createdAt: str = ""
    updatedAt: str = ""
    isActive: bool = True


class PersonalizationSchema(BaseModel):
    """
    Tracks user's learning profile for personalization.
    Updated after every chat session.
    """
    userId: str
    examTarget: str
    weakTopics: List[str] = []      # topics user struggles with
    strongTopics: List[str] = []    # topics user knows well
    difficultyLevel: str = "medium" # easy / medium / hard
    totalSessions: int = 0
    lastActive: str = ""