# main.py
import os
import io
import json
import glob
import zipfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import torch
import transformers  # Needed by HunyuanVideo
from diffusers import HunyuanVideoPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video  # For saving MP4s
from PIL import Image
# PngInfo is no longer needed
# from PIL.PngImagePlugin import PngInfo 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

# --- Configuration & Model Loading ---

APP_TITLE = "HunyuanVideo API"
MODEL_ID = "tencent/HunyuanVideo"
DEFAULT_NEG = "blurry, low-res, overexposed, extra fingers, deformed, text, watermark, logo"
OUTPUT_DIR = "outputs"


@torch.inference_mode()
def load_pipelines(num_instances: int = 1):
    """Loads the ML model pipelines into memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Safety: collapse to 1 on non-CUDA
    if device != "cuda":
        num_instances = 1
        
    pipes = []
    for _ in range(num_instances):
        # Use HunyuanVideoPipeline instead of StableDiffusionPipeline
        pipe = HunyuanVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            # safety_checker=None, # HunyuanVideo doesn't have a safety_checker arg
            use_safetensors=True,
        )
        # The pipeline loads its own default scheduler, which is recommended.
        # We no longer need to manually set EulerAncestralDiscreteScheduler.
        pipe = pipe.to(device)
        pipes.append(pipe)
        
    print(f"✅ Loaded {len(pipes)} pipeline(s) on device '{device}'")
    return pipes, device

# Load the models when the application starts
PIPES, DEVICE = load_pipelines(num_instances=2) # Load 2 instances by default if GPU allows

# --- FastAPI App Initialization ---

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# --- Helper Functions ---

# This function is no longer needed as we are not saving metadata into PNGs
# def _png_bytes_with_meta(img: Image.Image, meta_dict: dict) -> bytes:
#     ...

def _run_pipe(pipe, prompt, negative_prompt, steps, guidance, width, height, num_frames, count, seed_offset, base_seed, device):
    gens = [torch.Generator(device=device).manual_seed(base_seed + seed_offset + i) for i in range(count)]
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            num_frames=num_frames,             # Pass the number of frames
            num_images_per_prompt=count,       # This is the batch size for videos
            generator=gens,
        )
    # The output format is a list of videos, where each video is a list of PIL frames
    return out.frames  # e.g., [ [vid1_frame1, vid1_frame2], [vid2_frame1, vid2_frame2] ]

# --- API Data Models (using Pydantic) ---

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = DEFAULT_NEG
    steps: int = 28
    guidance: float = 6.5
    width: int = 512
    height: int = 512
    num_frames: int = 16         # Add num_frames for video
    batch_size: int = 1          # Number of *videos* to generate (default to 1)
    seed: Optional[int] = None

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "API is running", "model_id": MODEL_ID}

@app.post("/generate")
async def generate_images(req: GenerationRequest):
    """The main endpoint to generate videos."""
    base_seed = req.seed if req.seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
    
    n = len(PIPES)
    counts = [req.batch_size // n + (1 if i < (req.batch_size % n) else 0) for i in range(n)]
    offsets = [sum(counts[:i]) for i in range(n)]
    
    videos_all = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(_run_pipe, pipe, req.prompt, req.negative_prompt, req.steps, req.guidance,
                         req.width, req.height, req.num_frames, count, offset, base_seed, DEVICE)
                for pipe, count, offset in zip(PIPES, counts, offsets) if count > 0]
        for f in as_completed(futs):
            videos_all.extend(f.result()) # videos_all is now List[List[PIL.Image]]

    # --- Save files and create response ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{ts}"
    run_dir = os.path.join(OUTPUT_DIR, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    manifest_path = os.path.join(run_dir, "manifest.jsonl")

    response_files = []
    with open(manifest_path, "w", encoding="utf-8") as mf:
        # Each 'video_frames' is a List[PIL.Image]
        for i, video_frames in enumerate(videos_all):
            seed_used = base_seed + i
            meta = {
                "file": f"video_{i+1:02d}.mp4", # Save as MP4
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "steps": req.steps, "cfg": req.guidance, "size": f"{req.width}x{req.height}",
                "num_frames": req.num_frames, # Add frame count to metadata
                "seed": seed_used, "model": MODEL_ID, "timestamp": ts
            }
            
            # Save as MP4 video using diffusers utility
            video_path = os.path.join(run_dir, meta["file"])
            export_to_video(video_frames, video_path, fps=10) # 10 FPS default
            
            # Write to manifest
            mf.write(json.dumps(meta) + "\n")
            
            # Add file info to the API response
            response_files.append({
                "url": f"/images/{run_folder_name}/{meta['file']}",
                "seed": seed_used,
                "metadata": meta,
            })
            
    return {"base_seed": base_seed, "images": response_files, "run_folder": run_folder_name}


@app.get("/gallery")
async def get_gallery_data():
    """Replicates the logic from 1_Past_Images.py to find all videos/images."""
    run_dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "run_*")), key=os.path.getmtime, reverse=True)
    
    records = []
    for run_dir in run_dirs:
        run_folder_name = os.path.basename(run_dir)
        manifest = os.path.join(run_dir, "manifest.jsonl")
        if os.path.isfile(manifest):
            with open(manifest, "r", encoding="utf-8") as mf:
                for line in mf:
                    try:
                        rec = json.loads(line)
                        rec['url'] = f"/images/{run_folder_name}/{rec['file']}"
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
    return records