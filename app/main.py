import os
import json
import glob
import torch
import asyncio
import subprocess
import shutil
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# --- Configuration ---
APP_TITLE = "Lustify SDXL API"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Civitai Download Config
# Get your API key at https://civitai.com/user/account (Settings > API Keys)
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "YOUR_API_KEY_HERE") 
# Replace with the exact download URL from Civitai (Right-click "Download" -> Copy Link Address)
MODEL_DOWNLOAD_URL = "https://civitai.com/api/download/models/2155386?type=Model&format=SafeTensor&size=pruned&fp=fp16" 
MODEL_PATH = "models/lustify_sdxl.safetensors"

DEFAULT_NEG = "blurry, low-res, overexposed, extra fingers, deformed, text, watermark, logo"

# --- Global State ---
GALLERY_CACHE = []
CACHE_INITIALIZED = False

# --- Download Logic ---
def ensure_model_exists():
    """Checks if the model exists, and uses aria2c to download it rapidly if it doesn't."""
    if os.path.exists(MODEL_PATH):
        return # Model is already here, skip download

    print(f"⚠️ Model not found locally. Starting multi-threaded download via aria2c...")
    
    if not CIVITAI_API_KEY or CIVITAI_API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("Cannot download model: CIVITAI_API_KEY is missing. Please generate one in your Civitai account settings or set the environment variable.")

    if not shutil.which("aria2c"):
        raise FileNotFoundError(
            "aria2c is not installed or not in your system PATH. "
            "Please install it (e.g., 'sudo apt install aria2' on Linux or 'winget install aria2' on Windows)."
        )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    cmd = [
        "aria2c",
        "--console-log-level=warn",
        "--summary-interval=5",
        "-x", "16",
        "-s", "16",
        "-j", "16",
        "--auto-file-renaming=false",
        f"--header=Authorization: Bearer {CIVITAI_API_KEY}",
        "-d", os.path.dirname(MODEL_PATH),
        "-o", os.path.basename(MODEL_PATH),
        MODEL_DOWNLOAD_URL
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Download complete! Proceeding to load the model...")
    except subprocess.CalledProcessError as e:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(MODEL_PATH + ".aria2"):
            os.remove(MODEL_PATH + ".aria2")
        raise Exception(f"aria2c download failed with error code {e.returncode}.")

# --- Model Management ---
class ModelManager:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def load_model(self):
        # 1. Download the model if it's missing
        ensure_model_exists()
        
        print(f"🚀 Loading SDXL model on {self.device}...")
        
        # 2. Load the pipeline from the single safetensors file
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH,
            torch_dtype=self.dtype,
            use_safetensors=True
        )
        
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        
        # 3. Apply VRAM and Speed Optimizations
        if self.device == "cuda":
            # Speeds up memory access by reorganizing tensor structure
            pipe.unet.to(memory_format=torch.channels_last)
            
            # Reduces memory footprint during attention calculations
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ xformers enabled.")
            except Exception:
                print("⚠️ xformers not available. Relying on default PyTorch SDPA.")
                
        self.pipe = pipe
        print(f"✅ Ready.")

manager = ModelManager()

# --- FastAPI Setup ---
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# --- Schemas ---
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = DEFAULT_NEG
    steps: int = Field(default=28, ge=1, le=100)
    guidance: float = Field(default=6.5, ge=1.0, le=20.0)
    width: int = 1024  # SDXL native resolution
    height: int = 1024 # SDXL native resolution
    batch_size: int = Field(default=1, ge=1, le=8) # Configured for 22GB VRAM
    seed: Optional[int] = None

# --- Core Logic ---
def save_image_with_metadata(img, meta, run_dir):
    pnginfo = PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    
    file_path = os.path.join(run_dir, meta["file"])
    img.save(file_path, format="PNG", pnginfo=pnginfo)
    return file_path

def generate_images_sync(req_dict, base_seed):
    """Synchronous generation function offloaded to a thread."""
    generators = [
        torch.Generator(device=manager.device).manual_seed(base_seed + i) 
        for i in range(req_dict['batch_size'])
    ]
    
    with torch.inference_mode():
        output = manager.pipe(
            prompt=req_dict['prompt'],
            negative_prompt=req_dict['negative_prompt'],
            num_inference_steps=req_dict['steps'],
            guidance_scale=req_dict['guidance'],
            width=req_dict['width'],
            height=req_dict['height'],
            num_images_per_prompt=req_dict['batch_size'],
            generator=generators,
        )
    return output.images

def save_and_cache_background(images, req_dict, base_seed, ts, run_dir, run_folder):
    """Background task to handle disk I/O and cache updates."""
    global GALLERY_CACHE
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    
    with open(manifest_path, "a") as f:
        for i, img in enumerate(images):
            meta = {
                "file": f"image_{i}.png",
                "seed": base_seed + i,
                "prompt": req_dict['prompt'],
                "params": req_dict,
                "timestamp": ts,
                "url": f"/images/{run_folder}/image_{i}.png"
            }
            save_image_with_metadata(img, meta, run_dir)
            f.write(json.dumps(meta) + "\n")
            GALLERY_CACHE.insert(0, meta)

# --- Endpoints ---
@app.on_event("startup")
async def startup_event():
    manager.load_model()
    # Pre-warm the gallery cache
    _ = await get_gallery()

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "device": manager.device,
        "vram_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB" if manager.device == "cuda" else "N/A"
    }

@app.post("/generate")
async def generate(req: GenerationRequest, background_tasks: BackgroundTasks):
    base_seed = req.seed if req.seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
    
    try:
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None, generate_images_sync, req.dict(), base_seed
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"run_{ts}_{base_seed}"
        run_dir = os.path.join(OUTPUT_DIR, run_folder)
        os.makedirs(run_dir, exist_ok=True)
        
        response_data = []
        for i in range(len(images)):
            response_data.append({
                "url": f"/images/{run_folder}/image_{i}.png",
                "seed": base_seed + i
            })

        background_tasks.add_task(
            save_and_cache_background, 
            images, req.dict(), base_seed, ts, run_dir, run_folder
        )

        return {"status": "success", "images": response_data, "folder": run_folder}

    except Exception as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gallery")
async def get_gallery():
    global GALLERY_CACHE, CACHE_INITIALIZED
    
    if CACHE_INITIALIZED:
        return GALLERY_CACHE

    all_records = []
    manifests = glob.glob(os.path.join(OUTPUT_DIR, "**/manifest.jsonl"), recursive=True)
    
    for manifest in manifests:
        folder = os.path.basename(os.path.dirname(manifest))
        with open(manifest, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "url" not in data:
                        data["url"] = f"/images/{folder}/{data['file']}"
                    all_records.append(data)
                except json.JSONDecodeError:
                    continue
                
    GALLERY_CACHE = sorted(all_records, key=lambda x: x.get("timestamp", ""), reverse=True)
    CACHE_INITIALIZED = True
    
    return GALLERY_CACHE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)