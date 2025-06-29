from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    session_id: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int

class DocumentChunk(BaseModel):
    content: str
    metadata: dict
    similarity_score: Optional[float] = None