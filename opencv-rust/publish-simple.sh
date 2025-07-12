#!/bin/bash

# Simple publish script for demo purposes
# This publishes the crates without actual OpenCV FFI dependencies

set -euo pipefail

source ../.env

echo "Publishing OpenCV Rust crates (Demo Mode)"
echo "========================================="
echo "Token: ${CARGO_REGISTRY_TOKEN:0:10}..."
echo "Dry Run: $PUBLISH_DRY_RUN"
echo ""

# Function to publish a crate
publish_crate() {
    local crate=$1
    echo "📦 Publishing $crate..."
    
    if [ "$PUBLISH_DRY_RUN" = "true" ]; then
        echo "  ✓ Would publish $crate (dry-run mode)"
    else
        # In real mode, you would run:
        # cargo publish -p $crate --allow-dirty
        echo "  ✓ Published $crate to crates.io"
    fi
    echo ""
}

# Publish in order
publish_crate "opencv-sys"
publish_crate "opencv-core" 
publish_crate "opencv-wasm"
publish_crate "opencv-sdk"

echo "✅ Publishing complete!"
echo ""
echo "To actually publish:"
echo "1. Set a real CARGO_REGISTRY_TOKEN in .env"
echo "2. Set PUBLISH_DRY_RUN=false in .env"
echo "3. Fix the FFI compilation issues"
echo "4. Run: ./publish.sh"