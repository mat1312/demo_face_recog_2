from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import os
import time
import io
import base64
import uvicorn
from typing import List, Dict, Any, Optional
import uuid

# Create FastAPI app
app = FastAPI(
    title="🇮🇹 Meloni Detector",
    description="API for detecting Giorgia Meloni in images",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function for face recognition
def recognize_meloni(image, known_encodings, tolerance=0.58):
    # Search for all faces in the image
    face_locations = face_recognition.face_locations(image)
    
    # If no faces, return original image
    if not face_locations:
        return Image.fromarray(image), [], "Nessun volto trovato nell'immagine."
    
    # Get encodings for all found faces
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Create PIL image for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Track Meloni faces
    meloni_faces = []
    
    # Check each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known Meloni encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        
        # Calculate confidence
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        confidence = 1 - face_distances[best_match_index]
        
        # If it's a match
        if True in matches:
            # Draw green rectangle
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=4)
            
            # Add label
            label = f"Meloni ({confidence:.1%})"
            text_height = 20
            
            # Rectangle for text
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), 
                          fill=(0, 255, 0), outline=(0, 255, 0))
            
            # Text
            draw.text((left + 6, bottom - text_height - 5), label, fill=(0, 0, 0))
            
            # Add to Meloni faces list
            meloni_faces.append({
                "location": (top, right, bottom, left),
                "confidence": float(confidence)  # Convert numpy float to regular float for JSON
            })
        else:
            # Draw red rectangle for non-Meloni
            draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)
    
    # Prepare result message
    if meloni_faces:
        result_message = f"Trovate {len(meloni_faces)} istanze di Meloni nell'immagine!"
    else:
        result_message = "Meloni non trovata nell'immagine."
    
    return pil_image, meloni_faces, result_message

# Load Meloni face encodings
def load_meloni_encodings(folder_path="meloni_images"):
    encodings = []
    image_count = 0
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return encodings, image_count
    
    # Load all images in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image_count += 1
            
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    encodings.append(face_encodings[0])
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                
    return encodings, image_count

# Cache for encodings
MELONI_ENCODINGS = None
IMAGE_COUNT = 0

# Load encodings on startup
@app.on_event("startup")
async def startup_event():
    global MELONI_ENCODINGS, IMAGE_COUNT
    MELONI_ENCODINGS, IMAGE_COUNT = load_meloni_encodings()
    if len(MELONI_ENCODINGS) == 0:
        print("Warning: No valid images found in 'meloni_images' folder")

# API status endpoint
@app.get("/api/status")
async def get_status():
    global MELONI_ENCODINGS, IMAGE_COUNT
    return {
        "status": "online",
        "encodings": len(MELONI_ENCODINGS),
        "reference_images": IMAGE_COUNT
    }

# Demo analysis endpoint
@app.get("/api/demo")
async def analyze_demo(tolerance: float = 0.58):
    global MELONI_ENCODINGS
    
    if not MELONI_ENCODINGS:
        raise HTTPException(status_code=500, detail="No Meloni reference images found")
    
    demo_path = "foto_gruppo.jpg"
    if not os.path.exists(demo_path):
        raise HTTPException(status_code=404, detail="Demo image not found")
    
    try:
        # Process the demo image
        image = face_recognition.load_image_file(demo_path)
        result_image, meloni_faces, message = recognize_meloni(image, MELONI_ENCODINGS, tolerance)
        
        # Save result image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}.jpg"
        result_image.save(result_filename)
        
        # Convert to base64 for response
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare response
        return {
            "found": len(meloni_faces) > 0,
            "message": message,
            "faces_count": len(meloni_faces),
            "meloni_faces": meloni_faces,
            "max_confidence": max([face["confidence"] for face in meloni_faces]) if meloni_faces else 0,
            "result_image": f"data:image/jpeg;base64,{img_str}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Upload and analyze endpoint
@app.post("/api/analyze")
async def analyze_image(image: Dict[str, Any], tolerance: Optional[float] = 0.58):
    global MELONI_ENCODINGS
    
    if not MELONI_ENCODINGS:
        raise HTTPException(status_code=500, detail="No Meloni reference images found")
    
    try:
        # Extract the base64 image
        if "image" not in image:
            raise HTTPException(status_code=400, detail="Image data not provided")
        
        image_data = image["image"]
        # Remove data:image/jpeg;base64, prefix if present
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)
        
        # Process the image
        result_image, meloni_faces, message = recognize_meloni(img_array, MELONI_ENCODINGS, tolerance)
        
        # Save result image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}.jpg"
        result_image.save(result_filename)
        
        # Convert to base64 for response
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare response
        return {
            "found": len(meloni_faces) > 0,
            "message": message,
            "faces_count": len(meloni_faces),
            "meloni_faces": meloni_faces,
            "max_confidence": max([face["confidence"] for face in meloni_faces]) if meloni_faces else 0,
            "result_image": f"data:image/jpeg;base64,{img_str}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Main page redirect
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)