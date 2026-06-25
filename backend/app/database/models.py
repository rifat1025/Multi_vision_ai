from sqlalchemy import Column, Integer, String, DateTime,  ForeignKey, JSON
from sqlalchemy.sql import func
from .database import Base


class UploadedImage(Base):
    __tablename__ = "uploaded_images"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String(255))
    file_path = Column(String(500))

    uploaded_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)

    image_id = Column(Integer, ForeignKey("uploaded_images.id"))

    classification = Column(JSON)
    detected_objects = Column(JSON)
    caption = Column(String(1000))
    ocr_text = Column(String(5000))

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )