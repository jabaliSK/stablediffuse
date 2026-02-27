#!/bin/bash

# DreamPhotoGASM - Bare Metal Setup Script
# Run this script with sudo: sudo ./setup.sh

set -e # Exit immediately if a command exits with a non-zero status.

APP_DIR="/opt/dreamgasm"
SERVICE_NAME="dreamgasm"
NGINX_SITE="dreamgasm"

# 1. Check for Root
if [[ $EUID -ne 0 ]]; then
   echo "Error: This script must be run as root." 
   exit 1
fi

echo "--- Starting DreamPhotoGASM Setup ---"

# 2. Check for required files
if [[ ! -f "main.py" ]] || [[ ! -f "index.html" ]]; then
    echo "Error: main.py and index.html must be in the current directory."
    exit 1
fi

# 3. Install System Dependencies
echo "--- Installing System Dependencies ---"
apt-get update
apt-get install -y python3-pip python3-venv nginx acl

# 4. Setup Directories & Move Files
echo "--- Setting up Directory Structure ---"
mkdir -p "$APP_DIR/generated_images"
cp main.py "$APP_DIR/"
cp index.html "$APP_DIR/"

# 5. Setup Python Virtual Environment
echo "--- Setting up Python VENV (This may take a few minutes) ---"
cd "$APP_DIR"
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
fi

# Activate and Install
source venv/bin/activate
pip install --upgrade pip
# Installing dependencies
pip install fastapi uvicorn python-multipart diffusers transformers accelerate torch safetensors pydantic

# 6. Create Systemd Service
echo "--- Configuring Systemd Service ---"
cat <<EOF > /etc/systemd/system/${SERVICE_NAME}.service
[Unit]
Description=DreamPhotoGASM API Server
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
Environment="HUGGINGFACE_HUB_CACHE=$APP_DIR/.cache"
ExecStart=$APP_DIR/venv/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Reload Systemd
systemctl daemon-reload

# 7. Create Nginx Configuration
echo "--- Configuring Nginx ---"
cat <<EOF > /etc/nginx/sites-available/$NGINX_SITE
server {
    listen 80;
    server_name _;

    # Serve UI
    location / {
        root $APP_DIR;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    # Serve Images Directly
    location /images/ {
        alias $APP_DIR/generated_images/;
        autoindex off;
    }

    # Proxy API
    location /v1/ {
        proxy_pass http://127.0.0.1:8000/v1/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Proxy Health
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }

    # Proxy History
    location /history {
        proxy_pass http://127.0.0.1:8000/history;
    }
}
EOF

# Enable Nginx Site
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/$NGINX_SITE /etc/nginx/sites-enabled/
systemctl restart nginx

# 8. Start Application Service
echo "--- Starting Application Service ---"
systemctl enable $SERVICE_NAME
systemctl restart $SERVICE_NAME

# 9. Permissions
# Ensure the www-data (Nginx) user can read the images if needed, 
# though root owns the process, Nginx needs read access.
echo "--- Fixing Permissions ---"
chmod 755 $APP_DIR
chmod -R 755 "$APP_DIR/generated_images"

echo "--- Setup Complete! ---"
echo "You can view the logs using: journalctl -u $SERVICE_NAME -f"
echo "Access the app at: http://$(hostname -I | cut -d' ' -f1)"