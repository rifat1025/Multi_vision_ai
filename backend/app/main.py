
import os
from fastapi import FastAPI, Depends, UploadFile, File, BackgroundTasks, HTTPException
from typing import List
from app.database import models, schemas
from app.database.database import Base, engine, SessionLocal
from app.utils import process_and_save_analysis, lifespan
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Multi-Model Vision AI Platform", lifespan=lifespan)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload/", response_model=schemas.ImageUploadResponse)
async def upload_image(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    db_image = models.UploadedImage(filename=file.filename, file_path=file_path)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    # Hand off image path to the background thread running our ML models
    background_tasks.add_task(process_and_save_analysis, db_image.id, file_path, SessionLocal)

    return db_image

@app.get("/images/", response_model = List[schemas. ImageUploadResponse])

async def list_all_images(skip : int=0, limit : int=10, db:Session=Depends(get_db)):
    images = db.query(models.UploadedImage).offset(skip).limit(limit).all()
    
    return images



@app.get("/images/{image_id}/analysis",response_model=schemas.AnalysisResultResponse)

async def get_images_analysis(image_id:int,db:Session=Depends(get_db)):
    image = db.query(models.UploadedImage).filter(models.UploadedImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()
    if not analysis:
        raise HTTPException(status_code=202, detail="Processing...")
        
    return analysis

#integrate llm point


# Initialize Groq LLM
llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
            )
    

@app.post("/images/{image_id}/chat", response_model=schemas.ChatResponse)


def chat_with_image(image_id: int, payload: schemas.ChatRequest, db: Session = Depends(get_db)):
    
    
    
    analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()
    if not analysis:
        raise HTTPException(
            status_code=404, 
            detail="Analysis data not found for this image. It might still be processing."
        )
    # 2. Format the database results into a context string for the LLM
    context = f"""
    Image Caption: {analysis.caption}
    Detected Objects: {analysis.detected_objects}
    Extracted Text (OCR): {analysis.ocr_text}
    """

    # 3. Create a LangChain Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an advanced Vision AI Assistant. You are given data extracted from an image (Caption, Detected Objects with boxes/confidence, and OCR text). Answer the user's question accurately using ONLY the provided context. If you don't know, say you don't know."),
        ("user", "Context from Image:\n{image_context}\n\nQuestion: {user_question}")
    ])
    # 4. Chain them together and invoke
    chain = prompt | llm
    
    try:
        ai_answer = chain.invoke({
            "image_context": context,
            "user_question": payload.question
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    return schemas.ChatResponse(
        image_id=image_id,
        question=payload.question,
        answer=ai_answer .content
    )
    
