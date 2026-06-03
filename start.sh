#!/bin/bash
set -e

BUNDLE_DIR="/app/projects_bundle"
VOLUME_DIR="/app/projects"

echo "[start] Bundle dir exists: $([ -d "$BUNDLE_DIR" ] && echo YES || echo NO)"
echo "[start] Volume dir contents: $(ls "$VOLUME_DIR" 2>/dev/null || echo EMPTY)"

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
else
    echo "[start] WARNING: projects_bundle not found — volume will be empty"
fi

echo "[start] Final projects on volume: $(ls "$VOLUME_DIR" 2>/dev/null || echo NONE)"

exec streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true
