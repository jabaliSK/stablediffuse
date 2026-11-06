import os

# Ensure Hugging Face always caches to /workspace to avoid filling the small /root partition
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.makedirs("/workspace/.cache/huggingface", exist_ok=True)

import json
import glob
import torch
import asyncio
import threading
import time
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
APP_TITLE = "FLUX.1 Dev API"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "black-forest-labs/FLUX.1-dev"

# --- Global State ---
GALLERY_CACHE = []
CACHE_INITIALIZED = False

# Continuous Generation Server State (survives mobile browser sleep/disconnect)
CONTINUOUS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()  # Prevents concurrent requests from colliding on GPU VRAM
CONTINUOUS_RUNNING = False
CONTINUOUS_STATS = {
    "active": False,
    "prompt": "",
    "total_generated": 0,
    "last_image": None,
    "started_at": None
}

# --- Model Management ---
class ModelManager:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def load_model(self):
        print(f"🚀 Loading FLUX.1-dev model {MODEL_ID} on {self.device}...")
        
        # Enable Ampere/Ada TensorFloat-32 (TF32) and CuDNN benchmark for massive speedups
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("⚡ TF32 matrix multiplication & CuDNN benchmark enabled.")
        
        # Load the pipeline from HuggingFace
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            dtype=self.dtype
        )
        
        if self.device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Detected GPU with {total_vram_gb:.1f} GB VRAM")
            
            # If less than 24GB, use CPU offload and VAE tiling
            if total_vram_gb < 24:
                pipe.enable_model_cpu_offload()
                if hasattr(pipe, "vae") and pipe.vae is not None:
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()
                print("✅ Model CPU offload & VAE tiling enabled (<24GB VRAM).")
            else:
                pipe.to("cuda")
                # On 24GB+ GPUs (like A6000 48GB), VAE tiling is disabled to eliminate 2-4s latency per image
                if hasattr(pipe, "vae") and pipe.vae is not None:
                    pipe.vae.disable_slicing()
                    pipe.vae.disable_tiling()
                print("✅ Model loaded fully into GPU VRAM (VAE tiling disabled for instant decode).")
        else:
            pipe.to("cpu")
            
        self.pipe = pipe
        print(f"✅ FLUX.1 Dev Ready.")

    def warmup(self):
        """Warm up CUDA kernels and pipeline at startup to eliminate first-request latency."""
        if not self.pipe or self.device != "cuda":
            return
        print("🔥 Warming up FLUX pipeline on GPU...")
        try:
            with torch.inference_mode():
                self.pipe(
                    prompt="warmup",
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    width=512,
                    height=512,
                    max_sequence_length=64,
                )
            torch.cuda.empty_cache()
            print("✅ GPU warmup complete.")
        except Exception as e:
            print(f"⚠️ GPU warmup skipped: {e}")

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
    batch_size: Optional[int] = Field(default=1, ge=1, le=12)
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
    """Generates images in parallel chunks and saves them to disk before responding."""
    global GALLERY_CACHE
    
    with GPU_LOCK:
        # Ensure dimensions are multiples of 16 (required for Flux DiT)
        width = (int(req_dict.get('width') or 1024) // 16) * 16
        height = (int(req_dict.get('height') or 1024) // 16) * 16
        steps = int(req_dict.get('steps') or 25)
        guidance = float(req_dict.get('guidance') if req_dict.get('guidance') is not None else 3.5)
        batch_size = max(1, min(12, int(req_dict.get('batch_size') or 1)))
        
        # On 48GB A6000, parallel batch of 2 cuts generation time by ~45% per image pair
        can_parallel = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory / (1024**3) >= 30)
        parallel_chunk = 2 if can_parallel else 1
        
        response_data = []
        manifest_path = os.path.join(run_dir, "manifest.jsonl")
        total_generated = 0
        
        with open(manifest_path, "a") as f:
            while total_generated < batch_size:
                current_chunk = min(parallel_chunk, batch_size - total_generated)
                chunk_seeds = [base_seed + total_generated + j for j in range(current_chunk)]
                
                with torch.inference_mode():
                    if current_chunk > 1:
                        generators = [torch.Generator(device="cpu").manual_seed(s) for s in chunk_seeds]
                        output = manager.pipe(
                            prompt=req_dict['prompt'],
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            width=width,
                            height=height,
                            max_sequence_length=512,
                            num_images_per_prompt=current_chunk,
                            generator=generators,
                        )
                        chunk_images = output.images
                    else:
                        generator = torch.Generator(device="cpu").manual_seed(chunk_seeds[0])
                        output = manager.pipe(
                            prompt=req_dict['prompt'],
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            width=width,
                            height=height,
                            max_sequence_length=512,
                            generator=generator,
                        )
                        chunk_images = output.images
                
                for j, img in enumerate(chunk_images):
                    img_idx = total_generated + j
                    seed_val = chunk_seeds[j]
                    meta = {
                        "file": f"image_{img_idx}.png",
                        "seed": seed_val,
                        "prompt": req_dict['prompt'],
                        "params": req_dict,
                        "timestamp": ts,
                        "url": f"/images/{run_folder}/image_{img_idx}.png"
                    }
                    save_image_with_metadata(img, meta, run_dir)
                    f.write(json.dumps(meta) + "\n")
                    f.flush()
                    GALLERY_CACHE.insert(0, meta)
                    
                    response_data.append({
                        "url": f"/images/{run_folder}/image_{img_idx}.png",
                        "seed": seed_val
                    })
                
                total_generated += current_chunk
                
        return response_data

def continuous_worker(params: dict):
    """Server-side worker: keeps generating images on GPU even if mobile client disconnects."""
    global CONTINUOUS_RUNNING, CONTINUOUS_STATS
    print(f"🔄 [Background Worker] Continuous generation started for prompt: {params.get('prompt')[:40]}...")
    
    while True:
        with CONTINUOUS_LOCK:
            if not CONTINUOUS_RUNNING:
                break
                
        try:
            base_seed = torch.randint(0, 2**31 - 1, (1,)).item()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_folder = f"run_{ts}_{base_seed}"
            run_dir = os.path.join(OUTPUT_DIR, run_folder)
            os.makedirs(run_dir, exist_ok=True)
            
            # Generate 1 or 2 images per continuous step
            chunk_params = dict(params)
            chunk_params["batch_size"] = min(2, int(params.get("batch_size") or 1))
            
            results = generate_and_save_sync(chunk_params, base_seed, ts, run_dir, run_folder)
            
            with CONTINUOUS_LOCK:
                if results:
                    CONTINUOUS_STATS["total_generated"] += len(results)
                    CONTINUOUS_STATS["last_image"] = results[-1]["url"]
                    
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ [Background Worker] Error during continuous generation: {e}")
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
            
    print("🛑 [Background Worker] Continuous generation stopped.")

# --- Endpoints ---
@app.on_event("startup")
async def startup_event():
    manager.load_model()
    manager.warmup()
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
        if "out of memory" in str(e).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/continuous/start")
def start_continuous(req: GenerationRequest):
    global CONTINUOUS_RUNNING, CONTINUOUS_STATS
    with CONTINUOUS_LOCK:
        if CONTINUOUS_RUNNING:
            return {"status": "already_running", "stats": CONTINUOUS_STATS}
        
        CONTINUOUS_RUNNING = True
        CONTINUOUS_STATS = {
            "active": True,
            "prompt": req.prompt,
            "total_generated": 0,
            "last_image": None,
            "started_at": datetime.now().isoformat()
        }
        
        req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        thread = threading.Thread(target=continuous_worker, args=(req_dict,), daemon=True)
        thread.start()
        
    return {"status": "started", "stats": CONTINUOUS_STATS}

@app.post("/continuous/stop")
def stop_continuous():
    global CONTINUOUS_RUNNING, CONTINUOUS_STATS
    with CONTINUOUS_LOCK:
        CONTINUOUS_RUNNING = False
        CONTINUOUS_STATS["active"] = False
    return {"status": "stopped", "stats": CONTINUOUS_STATS}

@app.get("/continuous/status")
def get_continuous_status():
    with CONTINUOUS_LOCK:
        return CONTINUOUS_STATS

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