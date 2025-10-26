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
# --- MODIFIED: Added StableDiffusionImg2ImgPipeline ---
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# --- MODIFIED: Added File, Form, and UploadFile for the new endpoint ---
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

# --- Configuration & Model Loading ---

APP_TITLE = "epiCPhotoGasm API"
MODEL_ID = "Yntec/epiCPhotoGasm"
DEFAULT_NEG = "blurry, low-res, overexposed, extra fingers, deformed, text, watermark, logo"
OUTPUT_DIR = "outputs"

# --- MODIFIED: This function now loads BOTH Txt2Img and Img2Img pipelines ---
@torch.inference_mode()
def load_pipelines(num_instances: int = 1):
    """Loads the ML model pipelines into memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    if device != "cuda":
        num_instances = 1
        
    txt2img_pipes = []
    img2img_pipes = []
    
    for _ in range(num_instances):
        # 1. Load the base Txt2Img pipeline
        pipe_txt = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            use_safetensors=True,
        )
        pipe_txt.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_txt.scheduler.config)
        pipe_txt = pipe_txt.to(device)
        txt2img_pipes.append(pipe_txt)
        
        # 2. Create the Img2Img pipeline from the Txt2Img components
        # This is memory-efficient as it re-uses the loaded weights
        pipe_img = StableDiffusionImg2ImgPipeline(**pipe_txt.components)
        img2img_pipes.append(pipe_img)
        
    print(f"✅ Loaded {len(txt2img_pipes)} Txt2Img and {len(img2img_pipes)} Img2Img pipeline(s) on device '{device}'")
    return txt2img_pipes, img2img_pipes, device

# --- MODIFIED: Load both sets of models ---
TXT2IMG_PIPES, IMG2IMG_PIPES, DEVICE = load_pipelines(num_instances=2)

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

def _png_bytes_with_meta(img: Image.Image, meta_dict: dict) -> bytes:
    buf = io.BytesIO()
    pnginfo = PngInfo()
    for k, v in meta_dict.items():
        pnginfo.add_text(str(k), str(v))
    img.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()

# Helper for Txt2Img (Original)
def _run_pipe(pipe, prompt, negative_prompt, steps, guidance, width, height, count, seed_offset, base_seed, device):
    gens = [torch.Generator(device=device).manual_seed(base_seed + seed_offset + i) for i in range(count)]
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            num_images_per_prompt=count,
            generator=gens,
        )
    return out.images

# --- NEW: Helper for Img2Img ---
def _run_pipe_img2img(pipe, prompt, negative_prompt, steps, guidance, init_image, strength, count, seed_offset, base_seed, device):
    gens = [torch.Generator(device=device).manual_seed(base_seed + seed_offset + i) for i in range(count)]
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=count,
            generator=gens,
        )
    return out.images

# --- API Data Models (for Txt2Img) ---

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = DEFAULT_NEG
    steps: int = 28
    guidance: float = 6.5
    width: int = 512
    height: int = 512
    batch_size: int = 2
    seed: Optional[int] = None

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "API is running", "model_id": MODEL_ID}

# --- MODIFIED: Txt2Img endpoint now uses TXT2IMG_PIPES ---
@app.post("/generate")
async def generate_images(req: GenerationRequest):
    """The main endpoint to generate images from TEXT."""
    base_seed = req.seed if req.seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
    
    # --- MODIFIED: Use TXT2IMG_PIPES ---
    n = len(TXT2IMG_PIPES)
    counts = [req.batch_size // n + (1 if i < (req.batch_size % n) else 0) for i in range(n)]
    offsets = [sum(counts[:i]) for i in range(n)]
    
    images_all = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(_run_pipe, pipe, req.prompt, req.negative_prompt, req.steps, req.guidance,
                         req.width, req.height, count, offset, base_seed, DEVICE)
                # --- MODIFIED: Use TXT2IMG_PIPES ---
                for pipe, count, offset in zip(TXT2IMG_PIPES, counts, offsets) if count > 0]
        for f in as_completed(futs):
            images_all.extend(f.result())

    # --- Save files and create response ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{ts}"
    run_dir = os.path.join(OUTPUT_DIR, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    manifest_path = os.path.join(run_dir, "manifest.jsonl")

    response_files = []
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, img in enumerate(images_all):
            seed_used = base_seed + i
            meta = {
                "file": f"image_{i+1:02d}.png",
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "steps": req.steps, "cfg": req.guidance, "size": f"{req.width}x{req.height}",
                "seed": seed_used, "model": MODEL_ID, "timestamp": ts,
                "type": "txt2img"
            }
            
            png_bytes = _png_bytes_with_meta(img, meta)
            with open(os.path.join(run_dir, meta["file"]), "wb") as f:
                f.write(png_bytes)
            
            mf.write(json.dumps(meta) + "\n")
            
            response_files.append({
                "url": f"/images/{run_folder_name}/{meta['file']}",
                "seed": seed_used,
                "metadata": meta,
            })
            
    return {"base_seed": base_seed, "images": response_files, "run_folder": run_folder_name}


# --- NEW: Img2Img Endpoint ---
@app.post("/img2img")
async def generate_img2img(
    # This endpoint uses Form data instead of JSON to accept a file upload
    image: UploadFile = File(..., description="The initial image to modify."),
    prompt: str = Form(..., description="The prompt to guide the image generation."),
    negative_prompt: str = Form(DEFAULT_NEG),
    steps: int = Form(40, description="More steps (e.g., 40-50) are often good for Img2Img."),
    guidance: float = Form(7.0),
    strength: float = Form(0.7, ge=0.0, le=1.0, description="Controls how much the original image is changed. 0.0 = original, 1.0 = new image."),
    width: int = Form(512, description="Width to resize the input image to."),
    height: int = Form(512, description="Height to resize the input image to."),
    batch_size: int = Form(2),
    seed: Optional[int] = Form(None)
):
    """The main endpoint to generate images from an initial IMAGE."""
    
    # --- 1. Process the uploaded image ---
    try:
        image_bytes = await image.read()
        init_image_orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize to the target dimensions
        init_image = init_image_orig.resize((width, height), Image.Resampling.LANCZOS)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file provided: {e}")

    base_seed = seed if seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
    
    # --- 2. Parallel generation logic (using IMG2IMG_PIPES) ---
    n = len(IMG2IMG_PIPES)
    counts = [batch_size // n + (1 if i < (batch_size % n) else 0) for i in range(n)]
    offsets = [sum(counts[:i]) for i in range(n)]
    
    images_all = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(_run_pipe_img2img, pipe, prompt, negative_prompt, steps, guidance,
                         init_image, strength, count, offset, base_seed, DEVICE)
                for pipe, count, offset in zip(IMG2IMG_PIPES, counts, offsets) if count > 0]
        for f in as_completed(futs):
            images_all.extend(f.result())

    # --- 3. Save files and create response (similar to /generate) ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{ts}"
    run_dir = os.path.join(OUTPUT_DIR, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    manifest_path = os.path.join(run_dir, "manifest.jsonl")

    # Save the resized input image for reference
    init_image.save(os.path.join(run_dir, "input_image.png"))

    response_files = []
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, img in enumerate(images_all):
            seed_used = base_seed + i
            meta = {
                "file": f"image_{i+1:02d}.png",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps, "cfg": guidance, "size": f"{width}x{height}",
                "strength": strength, # Added strength to metadata
                "seed": seed_used, "model": MODEL_ID, "timestamp": ts,
                "type": "img2img"
            }
            
            png_bytes = _png_bytes_with_meta(img, meta)
            with open(os.path.join(run_dir, meta["file"]), "wb") as f:
                f.write(png_bytes)
            
            mf.write(json.dumps(meta) + "\n")
            
            response_files.append({
                "url": f"/images/{run_folder_name}/{meta['file']}",
                "seed": seed_used,
                "metadata": meta,
            })
            
    return {"base_seed": base_seed, "images": response_files, "run_folder": run_folder_name, "input_image_url": f"/images/{run_folder_name}/input_image.png"}


@app.get("/gallery")
async def get_gallery_data():
    """Replicates the logic from 1_Past_Images.py to find all images."""
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
                        # Also add the input image if it exists
                        input_img_path = os.path.join(run_dir, "input_image.png")
                        if os.path.isfile(input_img_path):
                            rec['input_image_url'] = f"/images/{run_folder_name}/input_image.png"
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
    return records