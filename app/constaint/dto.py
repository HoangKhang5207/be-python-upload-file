from fastapi import UploadFile
from pydantic import BaseModel
from typing import Optional


class DocumentDTO(BaseModel):
    user_id: Optional[int] = 0
    category_id: int
    description: str
