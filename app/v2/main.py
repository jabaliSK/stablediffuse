import torch
import uvicorn
import base64
import io
import logging
import os
import json
import uuid
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dreamphotogasm_server")

app = FastAPI(title="DreamPhotoGASM API", description="OpenAI-compatible Image Generation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Server Side Storage Setup ---
IMAGES_DIR = "generated_images"
HISTORY_FILE = "history.json"

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# Mount the images directory to serve files statically
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# --- Configuration ---
MODEL_ID = "Yntec/DreamPhotoGASM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

logger.info(f"Loading model: {MODEL_ID} on {DEVICE}...")

# --- Load Model ---
pipe = None

def load_model():
    global pipe
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            use_safetensors=True
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(DEVICE)
        
        if DEVICE == "cuda":
            pipe.enable_attention_slicing()
            
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Could not load the diffusion model.")

if __name__ != "__main__":
    load_model()

# --- Pydantic Models ---

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="A text description of the desired image(s).")
    negative_prompt: Optional[str] = Field(default="", description="Text to exclude from the image.")
    n: int = Field(default=1, ge=1, le=4, description="Number of images to generate.")
    size: str = Field(default="512x512", description="Size of the generated images.")
    response_format: str = Field(default="url", description="Format: 'url' or 'b64_json'.") # Default changed to URL
    steps: int = Field(default=25, description="Number of inference steps.")
    guidance_scale: float = Field(default=7.5, description="Classifier Free Guidance scale.")

class ImageObject(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None 

class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageObject]

# --- Endpoints ---

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/health")
def health_check():
    return {"status": "online", "model": MODEL_ID, "device": DEVICE}

@app.get("/history")
def get_history():
    """Returns the list of generated images from server history."""
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        # Reverse to show newest first
        return history[::-1]
    except Exception as e:
        logger.error(f"Error reading history: {e}")
        return []

@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    global pipe
    if pipe is None:
        load_model()

    try:
        logger.info(f"Received request: {request.prompt[:50]}...")
        
        # Parse size
        width, height = 512, 512
        if "x" in request.size:
            try:
                w, h = request.size.split("x")
                width, height = int(w), int(h)
            except ValueError:
                pass 
        
        # Run Inference
        images = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=height,
            width=width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.n
        ).images

        response_data = []
        created_timestamp = int(time.time())

        # Load existing history
        with open(HISTORY_FILE, "r") as f:
            history_db = json.load(f)

        for img in images:
            # Generate unique filename
            image_id = str(uuid.uuid4())
            filename = f"{image_id}.png"
            filepath = os.path.join(IMAGES_DIR, filename)
            
            # Save to disk
            img.save(filepath, format="PNG")
            
            # Create URL
            image_url = f"/images/{filename}"
            
            # Update History DB
            record = {
                "id": image_id,
                "url": image_url,
                "prompt": request.prompt,
                "timestamp": created_timestamp,
                "size": request.size,
                "steps": request.steps,
                "guidance": request.guidance_scale
            }
            history_db.append(record)
            
            # Prepare response
            if request.response_format == "b64_json":
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                response_data.append(ImageObject(b64_json=img_str, url=image_url))
            else:
                response_data.append(ImageObject(url=image_url))

        # Save history back to file
        with open(HISTORY_FILE, "w") as f:
            json.dump(history_db, f)

        return ImageGenerationResponse(
            created=created_timestamp,
            data=response_data
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)