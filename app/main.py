from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import torch
import os
from ultralytics.nn.tasks import SegmentationModel

# Get the base directory of this script (main.py)
# This is now the 'app' folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow YOLO segmentation model class for safe loading
torch.serialization.add_safe_globals([SegmentationModel])

app = FastAPI()

# Setup templates directory (Looks inside BASE_DIR/templates, e.g., /app/templates)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- MODEL CONFIGURATION ---
# Paths look directly inside the 'app' folder (BASE_DIR)
MODEL_PATH_DISEASE = os.path.join(BASE_DIR, "disease_best.pt")
MODEL_PATH_NUMBER = os.path.join(BASE_DIR, "number_best.pt")

# Attempt to load models safely
model_disease = None
model_number = None

try:
    model_disease = YOLO(MODEL_PATH_DISEASE)
    print(f"INFO: Disease Model loaded successfully from {MODEL_PATH_DISEASE}")
except Exception as e:
    print(f"CRITICAL ERROR: Disease Model failed to load. Check path: {MODEL_PATH_DISEASE}. Error: {e}")
    
try:
    model_number = YOLO(MODEL_PATH_NUMBER)
    print(f"INFO: Number Model loaded successfully from {MODEL_PATH_NUMBER}")
except Exception as e:
    print(f"CRITICAL ERROR: Number Model failed to load. Check path: {MODEL_PATH_NUMBER}. Error: {e}")
    

DISEASE_COLOR_MAP = {
    "Impacted": (0, 255, 0),
    "Caries": (255, 0, 0),
    "Deep Caries": (255, 255, 0),
    "Periapical Lesion": (0, 255, 255)
}

NUMBER_COLOR_MAP = {
    "Molar": (144, 238, 144),
    "Premolar": (255, 0, 0),
    "Canine": (255, 255, 0),
    "Incisors": (238, 130, 238)
}
# ---------------------------

def process_image_and_predict(img: np.ndarray, model: YOLO, color_map: dict) -> dict:
    """Generic function to run prediction, annotate, and return packaged data."""
    
    # Check if model object is valid before attempting prediction
    if model is None or not hasattr(model, 'model'):
        # Fallback if model failed to load
        _, buffer_orig = cv2.imencode(".jpg", img)
        img_base64_orig = base64.b64encode(buffer_orig).decode("utf-8")
        
        return {
            "img_base64_orig": img_base64_orig,
            "img_base64_annot": img_base64_orig, # Return original image as annotated
            "detections": [("Model Error: Failed to load.", 0.0)],
            "label_counts": {"Error": 1},
            "total_objects": 1
        }
    
    # Run YOLO prediction - Threshold lowered to 0.25 for better detection visibility
    results = model.predict(source=img, save=False, conf=0.25) 
    result = results[0]
    annotated_img = img.copy()
    detections = []

    # Handle segmentation masks if available
    masks = result.masks.data.cpu().numpy() if hasattr(result, "masks") and result.masks is not None else []

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = color_map.get(label, (255, 255, 255)) 

        # Draw mask
        if len(masks) > i:
            mask = masks[i]
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[:, :] = color
            
            # Blend original image with the colored mask for transparency
            annotated_img = np.where(
                mask[:, :, None] == 1,
                cv2.addWeighted(annotated_img, 0.5, colored_mask, 0.5, 0),
                annotated_img
            )

        # Draw bounding box and label
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        detections.append((label, conf))

    # Convert images to base64 for HTML display
    _, buffer_orig = cv2.imencode(".jpg", img)
    _, buffer_annot = cv2.imencode(".jpg", annotated_img)
    img_base64_orig = base64.b64encode(buffer_orig).decode("utf-8")
    img_base64_annot = base64.b64encode(buffer_annot).decode("utf-8")

    # Count each label occurrence
    label_counts = {}
    for label, _ in detections:
        label_counts[label] = label_counts.get(label, 0) + 1

    total_objects = len(detections)
    
    return {
        "img_base64_orig": img_base64_orig,
        "img_base64_annot": img_base64_annot,
        "detections": detections,
        "label_counts": label_counts,
        "total_objects": total_objects
    }

# --- NEW WORKFLOW ROUTES ---

# 1. Root route: Shows the prediction choice page (home.html)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# 2. Disease path: Shows the disease upload form
@app.get("/disease", response_class=HTMLResponse)
async def disease_home(request: Request):
    return templates.TemplateResponse(
        "disease_index.html", 
        {"request": request}
    )

# 3. Number path: Shows the teeth number upload form
@app.get("/number", response_class=HTMLResponse)
async def number_home(request: Request):
    return templates.TemplateResponse(
        "number_index.html", 
        {"request": request}
    )

# 4. Handle Disease Prediction (POST request from disease_index.html)
@app.post("/predict/disease", response_class=HTMLResponse)
async def predict_disease(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results_data = process_image_and_predict(img, model_disease, DISEASE_COLOR_MAP)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "title": "Dental Disease Prediction Results",
        **results_data
    })

# 5. Handle Teeth Number Prediction (POST request from number_index.html)
@app.post("/predict/number", response_class=HTMLResponse)
async def predict_number(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results_data = process_image_and_predict(img, model_number, NUMBER_COLOR_MAP)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "title": "Teeth Numbering Prediction Results",
        **results_data
    })
