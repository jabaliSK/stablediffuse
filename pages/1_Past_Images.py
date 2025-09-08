

# FILE: pages/1_Past_Images.py
# Gallery / history page that loads images from disk (outputs/ by default), supports fullscreen + next/prev.

import os
import json
import glob
import streamlit as st
from PIL import Image

DEFAULT_OUT_DIR = "outputs"

st.set_page_config(page_title="Past Images", page_icon="🖼️", layout="wide")
st.title("Past Images")

# Sidebar controls
with st.sidebar:
    out_root = st.text_input("Output folder", value=DEFAULT_OUT_DIR,
                             help="Browse this folder for saved runs (created by the main app).")
    sort_order = st.selectbox("Sort runs by", ["Newest first", "Oldest first"], index=0)
    thumbs_per_row = st.slider("Thumbnails per row", 2, 6, 4)

# Utility: discover runs and images
if not os.path.isdir(out_root):
    st.info("No output folder found yet. Generate images in the main page first.")
    st.stop()

# Gather run folders
run_dirs = sorted([d for d in glob.glob(os.path.join(out_root, "run_*")) if os.path.isdir(d)],
                  key=lambda p: os.path.getmtime(p), reverse=(sort_order == "Newest first"))

if not run_dirs:
    st.info("No runs found. Generate some images on the main page.")
    st.stop()

# Flatten all images across runs (newest first by default)
records = []  # list of (img_path, meta_dict)
for run in run_dirs:
    # Try to read manifest for richer metadata
    manifest = os.path.join(run, "manifest.jsonl")
    meta_by_file = {}
    if os.path.isfile(manifest):
        try:
            with open(manifest, "r", encoding="utf-8") as mf:
                for line in mf:
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and "file" in rec:
                            meta_by_file[rec["file"]] = rec
                    except Exception:
                        pass
        except Exception:
            pass

    for png in sorted(glob.glob(os.path.join(run, "*.png"))):
        fname = os.path.basename(png)
        meta = meta_by_file.get(fname, {})
        records.append((png, meta))

if not records:
    st.info("No images found in output folder.")
    st.stop()

# Manage selection / fullscreen index
if "gallery_index" not in st.session_state:
    st.session_state.gallery_index = None

# Grid of thumbnails
cols = st.columns(thumbs_per_row)
for idx, (path, meta) in enumerate(records):
    col = cols[idx % thumbs_per_row]
    with col:
        try:
            img = Image.open(path)
            st.image(img, caption=os.path.basename(path), use_column_width=True)
        except Exception:
            st.write(os.path.basename(path))
        if st.button("View", key=f"view_{idx}"):
            st.session_state.gallery_index = idx

# Fullscreen viewer
if st.session_state.gallery_index is not None:
    idx = st.session_state.gallery_index
    path, meta = records[idx]
    try:
        img = Image.open(path)
    except Exception:
        st.error("Failed to open image.")
        img = None

    st.markdown("---")
    st.subheader(os.path.basename(path))
    if img is not None:
        st.image(img, use_column_width=True)

    # Metadata display (if available)
    if meta:
        with st.expander("Metadata"):
            for k, v in meta.items():
                st.write(f"**{k}**: {v}")

    nav1, nav2, nav3 = st.columns([1,1,1])
    with nav1:
        if st.button("⬅️ Previous"):
            st.session_state.gallery_index = max(0, idx - 1)
    with nav2:
        if st.button("Close ❌"):
            st.session_state.gallery_index = None
    with nav3:
        if st.button("Next ➡️"):
            st.session_state.gallery_index = min(len(records) - 1, idx + 1)
