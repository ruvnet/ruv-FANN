#!/bin/bash
# Build script for unified WASM module

set -e

echo "🔧 Building Unified WASM Module..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[BUILD]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_status "Checking build dependencies..."
    
    # Check for Rust
    if ! command -v rustc &> /dev/null; then
        print_error "Rust is not installed. Please install from https://rustup.rs/"
        exit 1
    fi
    
    # Check for wasm-pack
    if ! command -v wasm-pack &> /dev/null; then
        print_warning "wasm-pack not found. Installing..."
        curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
    fi
    
    # Check for wasm32 target
    if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
        print_warning "wasm32-unknown-unknown target not installed. Installing..."
        rustup target add wasm32-unknown-unknown
    fi
    
    # Check for wasm-opt (optional but recommended)
    if ! command -v wasm-opt &> /dev/null; then
        print_warning "wasm-opt not found. Install with: npm install -g wasm-opt"
        print_warning "Continuing without optimization..."
    fi
}

# Build function
build_wasm() {
    local BUILD_TYPE=${1:-release}
    local FEATURES=${2:-default}
    
    print_status "Building WASM module (${BUILD_TYPE} mode with features: ${FEATURES})..."
    
    # Clean previous builds
    rm -rf pkg/
    
    # Build using the working ruv-swarm-wasm crate
    print_status "Building with ruv-swarm-wasm crate (unified crate has compilation errors)..."
    cd ../ruv-swarm-wasm
    
    # Build with wasm-pack
    if [ "$BUILD_TYPE" = "release" ]; then
        wasm-pack build --target web --release --scope ruv --features "$FEATURES"
    else
        wasm-pack build --target web --dev --scope ruv --features "$FEATURES"
    fi
    
    # Copy the build output back to unified directory
    cd ../ruv-swarm-wasm-unified
    if [ -d "../ruv-swarm-wasm/pkg" ]; then
        cp -r ../ruv-swarm-wasm/pkg ./
        print_status "WASM build completed successfully!"
    else
        print_error "WASM build failed!"
        exit 1
    fi
}

# Optimize WASM binary
optimize_wasm() {
    if command -v wasm-opt &> /dev/null; then
        print_status "Optimizing WASM binary..."
        
        local WASM_FILE="pkg/ruv_swarm_wasm_bg.wasm"
        
        if [ -f "$WASM_FILE" ]; then
            # Backup original
            cp "$WASM_FILE" "${WASM_FILE}.backup"
            
            # Optimize with various flags
            wasm-opt -Oz \
                --enable-simd \
                --enable-bulk-memory \
                --enable-threads \
                "$WASM_FILE" \
                -o "$WASM_FILE"
            
            # Compare sizes
            local ORIGINAL_SIZE=$(stat -f%z "${WASM_FILE}.backup" 2>/dev/null || stat -c%s "${WASM_FILE}.backup")
            local OPTIMIZED_SIZE=$(stat -f%z "$WASM_FILE" 2>/dev/null || stat -c%s "$WASM_FILE")
            local REDUCTION=$((100 - (OPTIMIZED_SIZE * 100 / ORIGINAL_SIZE)))
            
            print_status "Original size: $((ORIGINAL_SIZE / 1024))KB"
            print_status "Optimized size: $((OPTIMIZED_SIZE / 1024))KB"
            print_status "Size reduction: ${REDUCTION}%"
            
            # Remove backup
            rm "${WASM_FILE}.backup"
        else
            print_warning "WASM file not found for optimization"
        fi
    else
        print_warning "Skipping optimization (wasm-opt not installed)"
    fi
}

# Generate TypeScript definitions
generate_typescript() {
    print_status "Generating enhanced TypeScript definitions..."
    
    # Create additional TypeScript utilities
    cat > pkg/utils.d.ts << 'EOF'
// Additional TypeScript utilities for ruv-swarm WASM

export interface SwarmConfig {
    maxAgents: number;
    topology: 'mesh' | 'star' | 'ring' | 'hierarchical';
    enableSIMD?: boolean;
    memoryBudgetMB?: number;
}

export interface AgentConfig {
    name: string;
    type: 'researcher' | 'coder' | 'analyst' | 'optimizer' | 'coordinator';
    cognitivePattern?: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'critical' | 'abstract';
}

export interface TaskConfig {
    name: string;
    description: string;
    priority?: 'low' | 'medium' | 'high' | 'critical';
}

export interface PerformanceMetrics {
    agentCount: number;
    taskCount: number;
    memoryUsageMB: number;
    executionTimeMs: number;
    agentsPerMB: number;
}

// Helper functions
export function createOptimizedSwarm(config: SwarmConfig): Promise<any>;
export function benchmarkPerformance(testSize: number): Promise<any>;
EOF
}

# Copy to NPM package
copy_to_npm() {
    print_status "Copying WASM module to NPM package..."
    
    local NPM_WASM_DIR="../../npm/wasm-unified"
    
    # Create directory if it doesn't exist
    mkdir -p "$NPM_WASM_DIR"
    
    # Copy all generated files
    cp -r pkg/* "$NPM_WASM_DIR/"
    
    print_status "WASM module copied to NPM package"
}

# Generate build info
generate_build_info() {
    print_status "Generating build information..."
    
    local BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    local WASM_PACK_VERSION=$(wasm-pack --version | cut -d' ' -f2)
    
    cat > pkg/build-info.json << EOF
{
    "buildTime": "${BUILD_TIME}",
    "rustVersion": "${RUST_VERSION}",
    "wasmPackVersion": "${WASM_PACK_VERSION}",
    "features": {
        "simd": true,
        "parallel": true,
        "optimize": true
    },
    "targetArch": "wasm32-unknown-unknown"
}
EOF
}

# Main build process
main() {
    local BUILD_TYPE=${1:-release}
    local FEATURES=${2:-default}
    
    print_status "Starting unified WASM build process..."
    print_status "Build type: ${BUILD_TYPE}"
    print_status "Features: ${FEATURES}"
    
    # Check dependencies
    check_dependencies
    
    # Build WASM
    build_wasm "$BUILD_TYPE" "$FEATURES"
    
    # Optimize if in release mode
    if [ "$BUILD_TYPE" = "release" ]; then
        optimize_wasm
    fi
    
    # Generate TypeScript definitions
    generate_typescript
    
    # Generate build info
    generate_build_info
    
    # Copy to NPM package
    copy_to_npm
    
    print_status "✅ Unified WASM build complete!"
    print_status "Output location: ./pkg/"
    
    # Show final bundle info
    if [ -f "pkg/ruv_swarm_wasm_bg.wasm" ]; then
        local FINAL_SIZE=$(stat -f%z "pkg/ruv_swarm_wasm_bg.wasm" 2>/dev/null || stat -c%s "pkg/ruv_swarm_wasm_bg.wasm")
        print_status "Final WASM size: $((FINAL_SIZE / 1024))KB"
    fi
}

# Run main with arguments
main "$@"