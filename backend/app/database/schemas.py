from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- DETECTION SCHEMAS ---
class BoundingBox(BaseModel):
    box: List[float]   # [xmin, ymin, xmax, ymax]
    label: str
    confidence: float

# --- RESPONSE SCHEMAS ---
class AnalysisResultResponse(BaseModel):
    id: int
    image_id: int
    classification: Dict[str, float]       # e.g., {"outdoor": 0.92, "sunny": 0.85}
    detected_objects: List[BoundingBox]   # Array of detected bounding boxes
    caption: str
    ocr_text: str
    created_at: datetime

    class Config:
        from_attributes = True

class ImageUploadResponse(BaseModel):
    id: int
    filename: str
    file_path: str
    uploaded_at: datetime
    # We can optionally nest the analysis if it's done synchronously
    analysis: Optional[AnalysisResultResponse] = None

    class Config:
        from_attributes = True
        
        
# for integrate llm
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    image_id: int
    question: str
    answer: str