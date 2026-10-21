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

from diffusers import FluxPipeline
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# --- Configuration ---
APP_TITLE = "FLUX.1 Dev Uncensored API"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_ID = os.environ.get("LORA_ID", "enhanceaiteam/Flux-Uncensored-V2")

# --- Global State ---
GALLERY_CACHE = []
CACHE_INITIALIZED = False

# --- Model Management ---
class ModelManager:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def load_model(self):
        print(f"🚀 Loading FLUX.1-dev model {MODEL_ID} on {self.device}...")
        
        # Load the pipeline from HuggingFace
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype
        )
        
        # Load the uncensored LoRA
        if LORA_ID:
            try:
                print(f"📦 Loading Uncensored LoRA: {LORA_ID}...")
                pipe.load_lora_weights(LORA_ID, adapter_name="uncensored")
                print("✅ Uncensored LoRA loaded successfully.")
            except Exception as e:
                print(f"⚠️ Could not load LoRA {LORA_ID}: {e}. Proceeding with base model.")
        
        if self.device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Detected GPU with {total_vram_gb:.1f} GB VRAM")
            
            # If less than 24GB, use CPU offload
            if total_vram_gb < 24:
                pipe.enable_model_cpu_offload()
                print("✅ Model CPU offload enabled (<24GB VRAM).")
            else:
                pipe.to("cuda")
                print("✅ Model loaded fully into GPU VRAM (optimal for A6000 / 24GB+).")
                
            # VAE slicing & tiling to prevent memory spikes on high-resolution latents
            if hasattr(pipe, "vae") and pipe.vae is not None:
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
        else:
            pipe.to("cpu")
            
        self.pipe = pipe
        print(f"✅ FLUX.1 Dev Ready.")

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
    steps: int = Field(default=25, ge=1, le=50) # FLUX.1-dev standard is 20-30 steps
    guidance: float = Field(default=3.5, ge=1.0, le=15.0) # FLUX.1-dev guidance scale (3.5 default)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    batch_size: int = Field(default=1, ge=1, le=10)
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0) # Uncensored LoRA strength
    seed: Optional[int] = None

# --- Core Logic ---
def save_image_with_metadata(img, meta, run_dir):
    pnginfo = PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    
    file_path = os.path.join(run_dir, meta["file"])
    img.save(file_path, format="PNG", pnginfo=pnginfo)
    return file_path

def generate_and_save_sync(req_dict, base_seed, ts, run_dir, run_folder):
    """Generates images and completely saves them to disk before responding."""
    global GALLERY_CACHE
    
    # Ensure dimensions are multiples of 16 (required for Flux DiT)
    width = (req_dict.get('width', 1024) // 16) * 16
    height = (req_dict.get('height', 1024) // 16) * 16
    lora_scale = float(req_dict.get('lora_scale', 1.0))
    
    response_data = []
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    
    with open(manifest_path, "a") as f:
        for i in range(req_dict.get('batch_size', 1)):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            with torch.inference_mode():
                call_kwargs = {
                    "prompt": req_dict['prompt'],
                    "num_inference_steps": req_dict.get('steps', 25),
                    "guidance_scale": req_dict.get('guidance', 3.5),
                    "width": width,
                    "height": height,
                    "max_sequence_length": 512, # 512 for FLUX.1-dev
                    "generator": generator,
                }
                
                # Apply LoRA scale if adapters are loaded
                if hasattr(manager.pipe, "get_active_adapters") and manager.pipe.get_active_adapters():
                    call_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
                
                output = manager.pipe(**call_kwargs)
                img = output.images[0]
                
            meta = {
                "file": f"image_{i}.png",
                "seed": current_seed,
                "prompt": req_dict['prompt'],
                "params": req_dict,
                "timestamp": ts,
                "url": f"/images/{run_folder}/image_{i}.png"
            }
            save_image_with_metadata(img, meta, run_dir)
            f.write(json.dumps(meta) + "\n")
            f.flush()
            GALLERY_CACHE.insert(0, meta)
            
            response_data.append({
                "url": f"/images/{run_folder}/image_{i}.png",
                "seed": current_seed
            })
            
    return response_data

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
async def generate(req: GenerationRequest):
    base_seed = req.seed if req.seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
    
    try:
        loop = asyncio.get_event_loop()
        req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"run_{ts}_{base_seed}"
        run_dir = os.path.join(OUTPUT_DIR, run_folder)
        os.makedirs(run_dir, exist_ok=True)

        response_data = await loop.run_in_executor(
            None, generate_and_save_sync, req_dict, base_seed, ts, run_dir, run_folder
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