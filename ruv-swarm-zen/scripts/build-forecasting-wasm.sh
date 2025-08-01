#!/bin/bash
# Build script for RUV Swarm with forecasting capabilities

set -e

echo "Building RUV Swarm WASM with Neural Forecasting..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo -e "${RED}Error: wasm-pack is not installed${NC}"
    echo "Install it with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Navigate to the project root
cd "$(dirname "$0")/.."

# Build the ML crate first
echo -e "${YELLOW}Building ruv-swarm-ml...${NC}"
cargo build --manifest-path crates/ruv-swarm-ml/Cargo.toml --target wasm32-unknown-unknown --features wasm

# Build the main WASM module with ML features
echo -e "${YELLOW}Building ruv-swarm-wasm with ML features...${NC}"
cd crates/ruv-swarm-wasm
wasm-pack build --target web --features ml

# Check if build was successful
if [ -d "pkg" ]; then
    echo -e "${GREEN}✓ WASM build successful!${NC}"
    echo -e "${GREEN}✓ Output in: crates/ruv-swarm-wasm/pkg/${NC}"
    
    # Show the generated files
    echo -e "\n${YELLOW}Generated files:${NC}"
    ls -la pkg/
    
    # Show module size
    WASM_SIZE=$(du -h pkg/ruv_swarm_wasm_bg.wasm | cut -f1)
    echo -e "\n${YELLOW}WASM module size: ${WASM_SIZE}${NC}"
    
    # Copy to npm package if it exists
    if [ -d "../../npm/wasm" ]; then
        echo -e "\n${YELLOW}Copying WASM files to npm package...${NC}"
        cp -r pkg/* ../../npm/wasm/
        echo -e "${GREEN}✓ Files copied to npm/wasm/${NC}"
    fi
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

# Run tests
echo -e "\n${YELLOW}Running forecasting tests...${NC}"
cd ../..
cargo test --manifest-path crates/ruv-swarm-ml/Cargo.toml

echo -e "\n${GREEN}✓ Build and tests complete!${NC}"
echo -e "\n${YELLOW}To use the forecasting features:${NC}"
echo "1. Import the WASM module in your JavaScript:"
echo "   import init, { WasmNeuralForecast, WasmEnsembleForecaster } from './pkg/ruv_swarm_wasm.js';"
echo "2. Initialize with: await init();"
echo "3. Create a forecasting instance: const forecast = new WasmNeuralForecast(50.0);"
echo ""
echo "See examples/browser/forecasting.html for a complete example."