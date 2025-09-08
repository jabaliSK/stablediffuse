# epic_updated_patched.py
# Streamlit SD (epiCPhotoGasm) app with optional *two* parallel pipelines for faster batches.
# Drop-in replacement for your original app; adds "Parallel instances" and worker pool.
# Tested with: torch >= 2.0, diffusers >= 0.25, streamlit >= 1.29
import streamlit as st

st.set_page_config(
    page_title="epiCPhotoGasm",  # or APP_TITLE if you prefer
    page_icon="🖼️",
    layout="wide"
)
import io
import os
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from packaging import version
import zipfile
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

APP_TITLE = "epiCPhotoGasm — Parallel"
MODEL_ID = "Yntec/epiCPhotoGasm"  # from your original file

DEFAULT_NEG = "blurry, low-res, overexposed, extra fingers, deformed, text, watermark, logo"


# ---------- Helpers ----------

def _detect_device_dtype() -> Tuple[str, torch.dtype, bool, bool]:
    prefer_cuda = torch.cuda.is_available()
    prefer_mps = (not prefer_cuda) and torch.backends.mps.is_available()
    device = "cuda" if prefer_cuda else ("mps" if prefer_mps else "cpu")
    dtype = torch.float16 if (prefer_cuda or prefer_mps) else torch.float32
    return device, dtype, prefer_cuda, prefer_mps


def _apply_performance_mode(pipe, performance_mode: str, prefer_cuda: bool):
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    if performance_mode in {"balanced", "memory-saver"}:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
    else:
        # turbo
        try:
            pipe.disable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.disable_vae_tiling()
        except Exception:
            pass

    # Optional compile for more speed; safe settings
    if performance_mode == "turbo" and prefer_cuda:
        if version.parse(torch.__version__) >= version.parse("2.1"):
            try:
                pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=False)
            except Exception:
                pass
    return pipe


@st.cache_resource(show_spinner=False)
def load_pipelines(
    model_id: str,
    scheduler_name: str,
    performance_mode: str,
    num_instances: int,
):
    """Create 1 or 2 independent pipelines."""
    device, dtype, prefer_cuda, prefer_mps = _detect_device_dtype()

    # Safety: collapse to 1 on non-CUDA or on small GPUs
    if device == "cuda":
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            if total < 14 * (1024**3):
                num_instances = min(num_instances, 1)
        except Exception:
            pass
    else:
        num_instances = 1

    pipes = []
    for _ in range(num_instances):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            use_safetensors=True,
        )
        # Swap scheduler
        if scheduler_name.lower().startswith("euler"):
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(device)
        pipe = _apply_performance_mode(pipe, performance_mode, prefer_cuda)

        if device == "cpu" and version.parse(torch.__version__) >= version.parse("2.0"):
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:
                pass

        pipes.append(pipe)

    return pipes, device


def _run_pipe(pipe, prompt, negative_prompt, steps, guidance, width, height, count, seed_offset, base_seed, device):
    # Per-image generators for deterministic batches
    if base_seed is not None and base_seed >= 0:
        gens = [torch.Generator(device=device).manual_seed(base_seed + seed_offset + i) for i in range(count)]
    else:
        gens = None

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=(negative_prompt or None),
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            num_images_per_prompt=count,
            generator=gens,
        )
    imgs = out.images
    seeds_used = [(base_seed + seed_offset + i) if (base_seed is not None and base_seed >= 0) else None
                  for i in range(count)]
    return imgs, seeds_used


def generate_parallel(pipes: List, prompt: str, negative_prompt: str,
                      steps: int, guidance: float, width: int, height: int,
                      batch_size: int, base_seed: Optional[int], device: str):
    n = max(1, len(pipes))
    counts = [batch_size // n + (1 if i < (batch_size % n) else 0) for i in range(n)]
    if sum(counts) == 0:
        return [], []

    # Seed offsets to allocate a contiguous seed range per worker
    offsets = []
    off = 0
    for c in counts:
        offsets.append(off)
        off += c

    images_all, seeds_all = [], []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = []
        for pipe, count, seed_offset in zip(pipes, counts, offsets):
            if count == 0:
                continue
            futs.append(ex.submit(
                _run_pipe, pipe, prompt, negative_prompt, steps, guidance,
                width, height, count, seed_offset, base_seed, device
            ))
        for f in as_completed(futs):
            imgs, seeds = f.result()
            images_all.extend(imgs)
            seeds_all.extend(seeds)

    return images_all, seeds_all


def _png_bytes_with_meta(img: Image.Image, meta_dict: dict) -> bytes:
    buf = io.BytesIO()
    pnginfo = PngInfo()
    for k, v in meta_dict.items():
        try:
            pnginfo.add_text(str(k), str(v))
        except Exception:
            pass
    img.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()


def _default_instances() -> int:
    device, *_ = _detect_device_dtype()
    if device != "cuda":
        return 1
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        return 2 if total >= 14 * (1024**3) else 1
    except Exception:
        return 1


# ---------- UI ----------

st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Model & Performance")
    model_id = st.text_input("Model", value=MODEL_ID)
    scheduler_name = st.selectbox("Scheduler", ["Euler a (fast, crisp)", "DPM++ (clean, stable)"], index=0)
    sched_key = "euler" if scheduler_name.lower().startswith("euler") else "dpmpp"
    performance_mode = st.selectbox("Performance mode", ["memory-saver", "balanced", "turbo"], index=1,
                                    help="memory-saver/balanced enable attention slicing and VAE tiling; turbo prefers speed")
    parallel_instances = st.select_slider("Parallel instances", options=[1, 2], value=_default_instances(),
                                          help="Run multiple SD pipelines in parallel on the GPU")
    batch_size = st.slider("Batch size", 1, 8, 2)
    steps = st.slider("Steps", 10, 60, 28, key="steps")
    guidance = st.slider("CFG", 1.0, 12.0, 6.5, 0.5, key="cfg")

    sizes = [(512, 512), (640, 640), (768, 768), (832, 832), (896, 896), (1024, 1024)]
    w_h = st.selectbox("Size", options=[f"{w}×{h}" for (w, h) in sizes], index=0)
    width, height = map(int, w_h.split("×"))

    seed_in = st.text_input("Seed (leave blank for random)", value="")
    try:
        seed = int(seed_in) if seed_in.strip() != "" else None
    except Exception:
        seed = None
    st.caption("Tip: With a fixed seed, each image uses seed+index for reproducibility.")

prompt = st.text_area("Prompt", value="", height=120, placeholder="A photorealistic portrait...")
negative_prompt = st.text_area("Negative prompt (optional)", value=DEFAULT_NEG, height=80)

# Preload/instantiate the pipelines (cached)
pipes, device = load_pipelines(model_id, sched_key, performance_mode, 2)

# Heads-up for very large settings with 2 instances
if device == "cuda" and parallel_instances == 2 and (width * height >= 1024 * 1024) and batch_size >= 2:
    st.info("Running 2 instances at ≥1MP with batch ≥2. If you hit OOM, reduce size, batch, or switch to 'memory-saver'.")

col1, col2 = st.columns([1, 1])
with col1:
    go = st.button("Generate", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if go:
    base_seed = seed
    if base_seed is None:
        # Randomize a base seed so users can re-use it later
        base_seed = torch.randint(0, 2**31 - 1, (1,)).item()
        st.toast(f"Random base seed: {base_seed}")

    images, seeds_used = generate_parallel(
        pipes, prompt, negative_prompt, steps, guidance, width, height,
        batch_size, base_seed, device
    )
    out_root = "outputs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    mf = open(manifest_path, "w", encoding="utf-8")

    saved_files = []
    for i, img in enumerate(images):
        meta = {
            "file": f"image_{i+1:02d}.png",
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "steps": steps,
            "cfg": guidance,
            "size": f"{width}x{height}",
            "seed": seeds_used[i],
            "scheduler": sched_key,
            "mode": performance_mode,
            "model": model_id,
            "device": device,
            "timestamp": ts,
            "index": i + 1,
        }
        png_bytes = _png_bytes_with_meta(img, meta)

        # Save to disk for gallery
        out_path = os.path.join(run_dir, meta["file"])
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        saved_files.append(out_path)

        # Keep showing in the UI (existing behavior)
        cap = f"Image {i+1}"
        if seeds_used[i] is not None:
            cap += f" • seed {seeds_used[i]}"
        st.image(img, caption=cap, use_column_width=True)

        # Record a manifest line for the gallery page
        import json as _json
        mf.write(_json.dumps(meta, ensure_ascii=False) + "\n")

    mf.close()
    st.success(f"Saved {len(saved_files)} image(s) to: {run_dir}")    
    # Display and collect PNG bytes
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images):
            cap = f"Image {i+1}"
            if seeds_used[i] is not None:
                cap += f" • seed {seeds_used[i]}"
            st.image(img, caption=cap, use_column_width=True)

            meta = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "steps": steps,
                "cfg": guidance,
                "size": f"{width}x{height}",
                "seed": seeds_used[i],
                "scheduler": sched_key,
                "mode": performance_mode,
            }
            png_bytes = _png_bytes_with_meta(img, meta)
            zf.writestr(f"image_{i+1:02d}.png", png_bytes)

    if len(images) > 1:
        st.download_button("Download all as ZIP",
                           data=zip_buf.getvalue(),
                           file_name="sd_batch.zip",
                           mime="application/zip")
