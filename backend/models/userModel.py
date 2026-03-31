from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    firstName: str
    lastName: str
    email: str
    role: str                        # "student"
    examTarget: str                  # "JEE" or "NEET"
    profileImageUrl: Optional[str] = None
    clerkId: str
    isActive: bool = True