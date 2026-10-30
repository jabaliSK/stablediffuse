#!/bin/bash
# ==============================================================================
# Vast.ai Total Reset & Clean Start Script
# Stops all processes, deletes repositories, frees disk cache, and prepares fresh state.
# ==============================================================================

echo "🛑 [1/5] Killing all running server and node processes..."
pkill -9 -f "uvicorn" || true
pkill -9 -f "fastapi" || true
pkill -9 -f "npm" || true
pkill -9 -f "vite" || true
pkill -9 -f "node" || true
pkill -9 -f "esbuild" || true
sleep 2

echo "🗑️  [2/5] Deleting project repositories and output directories..."
rm -rf /workspace/stablediffuse
rm -rf /workspace/stablediffuse-main
rm -rf /workspace/outputs
rm -rf /workspace/backend.log /workspace/frontend.log /workspace/nohup.out

echo "🧹 [3/5] Clearing Hugging Face model cache (frees up 30-40GB)..."
rm -rf /workspace/.cache/huggingface
rm -rf /root/.cache/huggingface
rm -rf ~/.cache/huggingface

echo "🧹 [4/5] Clearing temporary files, npm cache, pip cache, and apt cache..."
rm -rf ~/.cache/pip /root/.cache/pip
rm -rf ~/.npm /root/.npm
rm -rf /tmp/* /var/tmp/*
apt-get clean || true

echo "📊 [5/5] Checking available disk space after wipe:"
df -h /workspace 2>/dev/null || df -h /

echo ""
echo "=============================================================================="
echo "✅ Reset Complete! Your Vast instance is back to a pristine state."
echo ""
echo "To start completely fresh, run:"
echo "------------------------------------------------------------------------------"
echo "export HF_TOKEN=\"your_hf_token_here\""
echo "git clone -b FLUX https://github.com/jabaliSK/stablediffuse.git"
echo "chmod +x stablediffuse/app/commands.txt"
echo "./stablediffuse/app/commands.txt FLUX"
echo "=============================================================================="
