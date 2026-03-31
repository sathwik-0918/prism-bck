from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# like your userCommentSchema
class Message(BaseModel):
    role: str                        # "user" or "assistant"
    content: str
    sources: Optional[List[str]] = []
    timestamp: str

# like your articleSchema
class ChatSession(BaseModel):
    sessionId: str
    userId: str
    title: str
    messages: List[Message] = []
    createdAt: str
    isActive: bool = True