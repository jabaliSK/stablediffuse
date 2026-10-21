import os

# Ensure Hugging Face always caches to /workspace to avoid filling the small /root partition
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.makedirs("/workspace/.cache/huggingface", exist_ok=True)

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
LORA_ID = os.environ.get("LORA_ID", "kenerateai/Flux-uncensored")

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
            dtype=self.dtype
        )
        
        # Load the uncensored LoRA
        if LORA_ID:
            loaded = False
            for lora_spec in [
                {"repo": LORA_ID, "kwargs": {}},
                {"repo": "kenerateai/Flux-uncensored", "kwargs": {}},
                {"repo": "devsquad8338/Flux-Uncensored-V2-bucket", "kwargs": {"weight_name": "lora.safetensors"}},
            ]:
                try:
                    print(f"📦 Loading Uncensored LoRA: {lora_spec['repo']}...")
                    pipe.load_lora_weights(lora_spec['repo'], adapter_name="uncensored", **lora_spec['kwargs'])
                    print(f"✅ Uncensored LoRA ({lora_spec['repo']}) loaded successfully.")
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Could not load {lora_spec['repo']}: {e}")
            
            if not loaded:
                print("⚠️ No LoRA loaded. Proceeding with base FLUX.1-dev model.")
        
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
    steps: Optional[int] = 25
    guidance: Optional[float] = 3.5
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    batch_size: Optional[int] = 1
    lora_scale: Optional[float] = 1.0
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None

    class Config:
        extra = "ignore"

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
    width = (int(req_dict.get('width') or 1024) // 16) * 16
    height = (int(req_dict.get('height') or 1024) // 16) * 16
    steps = int(req_dict.get('steps') or 25)
    guidance = float(req_dict.get('guidance') if req_dict.get('guidance') is not None else 3.5)
    lora_scale = float(req_dict.get('lora_scale') if req_dict.get('lora_scale') is not None else 1.0)
    batch_size = max(1, int(req_dict.get('batch_size') or 1))
    
    response_data = []
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    
    with open(manifest_path, "a") as f:
        for i in range(batch_size):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            with torch.inference_mode():
                call_kwargs = {
                    "prompt": req_dict['prompt'],
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
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