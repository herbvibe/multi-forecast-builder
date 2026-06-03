#!/bin/bash
# Seed the persistent volume with bundled projects on first boot.
# The volume is mounted at /app/projects and starts empty, which would
# shadow the projects/ directory baked into the image.  Copy them over
# once so the app has its default projects available immediately.

BUNDLE_DIR="/app/projects_bundle"
VOLUME_DIR="/app/projects"

if [ -d "$BUNDLE_DIR" ]; then
    for proj in "$BUNDLE_DIR"/*/; do
        proj_id=$(basename "$proj")
        if [ ! -d "$VOLUME_DIR/$proj_id" ]; then
            echo "[start] Seeding project '$proj_id' into volume..."
            cp -r "$proj" "$VOLUME_DIR/$proj_id"
        else
            echo "[start] Project '$proj_id' already on volume, skipping."
        fi
    done
fi

exec streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true
