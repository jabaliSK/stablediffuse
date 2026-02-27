import os
import io
import json
import glob
import torch
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# --- Configuration ---
APP_TITLE = "epiCPhotoGasm API"
MODEL_ID = "Yntec/epiCPhotoGasm"
DEFAULT_NEG = "blurry, low-res, overexposed, extra fingers, deformed, text, watermark, logo"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Management ---
class ModelManager:
    def __init__(self):
        self.pipes = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = None

    def load_models(self, num_instances: int = 1):
        if self.device != "cuda":
            num_instances = 1
        
        print(f"🚀 Loading {num_instances} instance(s) on {self.device}...")
        for _ in range(num_instances):
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=self.dtype,
                use_safetensors=True,
                safety_checker=None 
            )
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(self.device)
            # Enable memory optimizations if using CUDA
            if self.device == "cuda":
                pipe.enable_attention_slicing()
            self.pipes.append(pipe)
        
        self.executor = ThreadPoolExecutor(max_workers=len(self.pipes))
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
    steps: int = Field(default=25, ge=1, le=100)
    guidance: float = Field(default=7.0, ge=1.0, le=20.0)
    width: int = 512
    height: int = 512
    batch_size: int = Field(default=1, ge=1, le=8)
    seed: Optional[int] = None

# --- Core Logic ---
def save_image_with_metadata(img, meta, run_dir):
    pnginfo = PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    
    file_path = os.path.join(run_dir, meta["file"])
    img.save(file_path, format="PNG", pnginfo=pnginfo)
    return file_path

def sync_generate(pipe, req_dict, count, offset, base_seed):
    """Synchronous generation function to be run in a thread."""
    generators = [
        torch.Generator(device=manager.device).manual_seed(base_seed + offset + i) 
        for i in range(count)
    ]
    
    with torch.inference_mode():
        output = pipe(
            prompt=req_dict['prompt'],
            negative_prompt=req_dict['negative_prompt'],
            num_inference_steps=req_dict['steps'],
            guidance_scale=req_dict['guidance'],
            width=req_dict['width'],
            height=req_dict['height'],
            num_images_per_prompt=count,
            generator=generators,
        )
    return output.images

# --- Endpoints ---
@app.on_event("startup")
async def startup_event():
    # Load 1 instance by default to save VRAM, increase if your GPU is beefy
    manager.load_models(num_instances=1)

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "device": manager.device,
        "instances": len(manager.pipes),
        "vram_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB" if manager.device == "cuda" else "N/A"
    }

@app.post("/generate")
async def generate(req: GenerationRequest):
    base_seed = req.seed if req.seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
    
    # Split work across pipelines
    n = len(manager.pipes)
    counts = [req.batch_size // n + (1 if i < (req.batch_size % n) else 0) for i in range(n)]
    offsets = [sum(counts[:i]) for i in range(n)]
    
    try:
        loop = asyncio.get_event_loop()
        tasks = []
        for i, pipe in enumerate(manager.pipes):
            if counts[i] > 0:
                tasks.append(loop.run_in_executor(
                    manager.executor, sync_generate, pipe, req.dict(), counts[i], offsets[i], base_seed
                ))
        
        results = await asyncio.gather(*tasks)
        all_images = [img for sublist in results for img in sublist]

        # Saving process
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"run_{ts}_{base_seed}"
        run_dir = os.path.join(OUTPUT_DIR, run_folder)
        os.makedirs(run_dir, exist_ok=True)
        
        response_data = []
        manifest_path = os.path.join(run_dir, "manifest.jsonl")

        with open(manifest_path, "a") as f:
            for i, img in enumerate(all_images):
                meta = {
                    "file": f"image_{i}.png",
                    "seed": base_seed + i,
                    "prompt": req.prompt,
                    "params": req.dict(),
                    "timestamp": ts
                }
                save_image_with_metadata(img, meta, run_dir)
                f.write(json.dumps(meta) + "\n")
                
                response_data.append({
                    "url": f"/images/{run_folder}/{meta['file']}",
                    "seed": meta["seed"]
                })

        return {"status": "success", "images": response_data, "folder": run_folder}

    except Exception as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gallery")
async def get_gallery():
    """Optimized gallery fetcher."""
    all_records = []
    # Find all manifest files
    manifests = glob.glob(os.path.join(OUTPUT_DIR, "**/manifest.jsonl"), recursive=True)
    
    for manifest in manifests:
        folder = os.path.basename(os.path.dirname(manifest))
        with open(manifest, "r") as f:
            for line in f:
                data = json.loads(line)
                data["url"] = f"/images/{folder}/{data['file']}"
                all_records.append(data)
                
    return sorted(all_records, key=lambda x: x.get("timestamp", ""), reverse=True)
