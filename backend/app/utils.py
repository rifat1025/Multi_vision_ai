# ML libraries for vision pipeline (placeholders for now)
from ultralytics import YOLO  # For YOLO
from PIL import Image  # For image processing
from transformers import pipeline as hf_pipeline  # For HuggingFace models
import easyocr  # For OCR
from fastapi import FastAPI
from app.database import models
import cv2
import os

# Global dictionary to keep models in memory
ml_models = {}

def lifespan(app: FastAPI):
    print("AI models in memory:")

    # 1. YOLOv8 Nano (Saves ~40MB)
    ml_models["yolo"] = YOLO("yolov8m.pt")
    
    # 2. GIT Base for Captioning (Saves ~300MB)
    ml_models["captioner"] = hf_pipeline("image-to-text", model="microsoft/git-base", framework="pt")
    
    # 3. MobileViT for Classification (Saves ~320MB!)
    ml_models["classifier"] = hf_pipeline("image-classification", model="apple/mobilevit-small", framework="pt")
    
    # 4. EasyOCR (Downloads ~100MB for English)
    ml_models["ocr"] = easyocr.Reader(['en'], gpu=False)
    
    print("All models successfully loaded into memory")

    yield

    ml_models.clear()

# --- Fake/Placeholder Vision Pipeline Functions ---
# Replace these later with your actual YOLO, HuggingFace, and OCR implementations
def run_vision_pipeline(image_path: str) -> dict:
    # 1. YOLO Detection -> returns list of dicts
    # 2. HF Transformers -> returns classification dict & caption string
    # 3. OCR Engine -> returns raw string
    return {
        "classification": {"scenery": 0.95, "nature": 0.88},
        "detected_objects": [{"box": [10, 20, 100, 200], "label": "tree", "confidence": 0.91}],
        "caption": "A beautiful green landscape under a blue sky.",
        "ocr_text": "Welcome to National Park"
    }

    
def process_and_save_analysis(image_id: int, file_path: str, db_session_factory):
    db = db_session_factory()
    try:
        pil_img = Image.open(file_path).convert("RGB")

        # 1. Object Detection (YOLOv8)
        yolo_results = ml_models["yolo"](file_path)[0]
        detected_objects = []
        for box in yolo_results.boxes:
            detected_objects.append({
                "box": [round(x, 2) for x in box.xyxy[0].tolist()],
                "label": yolo_results.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2)
            })

        # --- SAVE THE ANNOTATED IMAGE ---
        annotated_dir = "static/annotated"
        os.makedirs(annotated_dir, exist_ok=True)
        # Generate an annotated image array from YOLO results
        annotated_frame = yolo_results.plot() 
        annotated_path = os.path.join(annotated_dir, f"detected_{image_id}.jpg")
        cv2.imwrite(annotated_path, annotated_frame)
        # --------------------------------

        # 2. Image Captioning (BLIP)
        caption_output = ml_models["captioner"](pil_img)
        caption_text = caption_output[0]['generated_text']

        # 3. Classification (ViT)
        clf_output = ml_models["classifier"](pil_img)
        classification_dict = {item['label']: round(item['score'], 3) for item in clf_output}

        # 4. OCR Text Extraction (EasyOCR)
        ocr_results = ml_models["ocr"].readtext(file_path, detail=0)
        extracted_text = " ".join(ocr_results)

        # 5. Save everything to DB
        db_analysis = models.AnalysisResult(
            image_id=image_id,
            classification=classification_dict,
            detected_objects=detected_objects,
            caption=caption_text,
            ocr_text=extracted_text
        )
        db.add(db_analysis)
        db.commit()

    except Exception as e:
        print(f"Error processing vision pipeline for image {image_id}: {e}")
    finally:
        db.close()


# --- API ENDPOINTS ---