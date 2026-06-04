#!/bin/bash
set -e

BUNDLE_DIR="/app/projects_bundle"
VOLUME_DIR="/app/projects"
MODELS_URL="https://github.com/herbvibe/multi-forecast-builder/releases/download/v1.0-models/flensburg-models.tar.gz"

echo "[start] Bundle dir exists: $([ -d "$BUNDLE_DIR" ] && echo YES || echo NO)"
echo "[start] Volume dir contents: $(ls "$VOLUME_DIR" 2>/dev/null || echo EMPTY)"

# Seed config + data CSVs from the bundle (fast, always safe)
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

# Download pre-trained models if not already on the volume
MODELS_MARKER="$VOLUME_DIR/.models_seeded_v2"
if [ ! -f "$MODELS_MARKER" ]; then
    echo "[start] Downloading pre-trained models from GitHub release..."
    curl -L "$MODELS_URL" -o /tmp/m.tar.gz \
        && echo "[start] Download complete ($(du -sh /tmp/m.tar.gz | cut -f1))" \
        && tar -xzf /tmp/m.tar.gz -C /app \
        && echo "[start] Extraction complete." \
        && rm /tmp/m.tar.gz \
        && touch "$MODELS_MARKER" \
        && echo "[start] Models seeded successfully." \
        || echo "[start] WARNING: model download/extraction failed."
else
    echo "[start] Models already seeded (marker found), skipping download."
fi

echo "[start] Final projects on volume: $(ls "$VOLUME_DIR" 2>/dev/null || echo NONE)"

exec streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true
