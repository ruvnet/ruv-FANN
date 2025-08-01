#!/bin/bash
# Build orchestrator for multi-crate WASM compilation

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../../.."
UNIFIED_CRATE_DIR="$SCRIPT_DIR/.."
BUILD_CACHE_DIR="$UNIFIED_CRATE_DIR/.build-cache"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Build configuration
declare -A BUILD_CONFIGS=(
    ["core"]="wasm-config/core.toml"
    ["neural"]="wasm-config/neural.toml"
    ["forecasting"]="wasm-config/forecasting.toml"
)

# Initialize build environment
init_build_env() {
    log_info "Initializing build environment..."
    
    # Create build cache directory
    mkdir -p "$BUILD_CACHE_DIR"
    
    # Check Rust toolchain
    if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
        log_warning "Installing wasm32-unknown-unknown target..."
        rustup target add wasm32-unknown-unknown
    fi
    
    # Install required tools
    local TOOLS_NEEDED=()
    
    command -v wasm-pack &> /dev/null || TOOLS_NEEDED+=("wasm-pack")
    command -v wasm-opt &> /dev/null || TOOLS_NEEDED+=("wasm-opt")
    command -v wasm-bindgen &> /dev/null || TOOLS_NEEDED+=("wasm-bindgen-cli")
    
    if [ ${#TOOLS_NEEDED[@]} -gt 0 ]; then
        log_warning "Installing missing tools: ${TOOLS_NEEDED[*]}"
        
        for tool in "${TOOLS_NEEDED[@]}"; do
            case "$tool" in
                "wasm-pack")
                    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
                    ;;
                "wasm-opt")
                    npm install -g wasm-opt
                    ;;
                "wasm-bindgen-cli")
                    cargo install wasm-bindgen-cli
                    ;;
            esac
        done
    fi
    
    log_success "Build environment initialized"
}

# Build individual module
build_module() {
    local MODULE=$1
    local CONFIG_FILE="${BUILD_CONFIGS[$MODULE]}"
    
    log_info "Building $MODULE module with config: $CONFIG_FILE"
    
    # Load configuration
    if [ -f "$UNIFIED_CRATE_DIR/$CONFIG_FILE" ]; then
        # Extract features from TOML
        local FEATURES=$(grep -E "^default\s*=\s*\[" "$UNIFIED_CRATE_DIR/$CONFIG_FILE" | sed 's/.*\[\(.*\)\].*/\1/' | tr -d '"' | tr ',' ' ')
        log_info "Features for $MODULE: $FEATURES"
    else
        log_warning "Config file not found: $CONFIG_FILE, using defaults"
        local FEATURES="default"
    fi
    
    # Build with specific features
    cd "$UNIFIED_CRATE_DIR"
    
    # Clean previous module build
    rm -rf "pkg-$MODULE"
    
    # Build module
    wasm-pack build \
        --target web \
        --release \
        --out-dir "pkg-$MODULE" \
        --out-name "ruv_swarm_${MODULE}" \
        -- --features "$FEATURES"
    
    # Optimize if wasm-opt is available
    if command -v wasm-opt &> /dev/null; then
        optimize_module "$MODULE"
    fi
    
    log_success "Module $MODULE built successfully"
}

# Optimize WASM module
optimize_module() {
    local MODULE=$1
    local WASM_FILE="pkg-$MODULE/ruv_swarm_${MODULE}_bg.wasm"
    
    if [ -f "$WASM_FILE" ]; then
        log_info "Optimizing $MODULE module..."
        
        # Get optimization flags from config
        local OPT_FLAGS="-Oz --enable-simd"
        
        # Backup original
        cp "$WASM_FILE" "${WASM_FILE}.original"
        
        # Optimize
        wasm-opt $OPT_FLAGS "$WASM_FILE" -o "$WASM_FILE"
        
        # Report size reduction
        local ORIGINAL_SIZE=$(stat -c%s "${WASM_FILE}.original" 2>/dev/null || stat -f%z "${WASM_FILE}.original")
        local OPTIMIZED_SIZE=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")
        local REDUCTION=$((100 - (OPTIMIZED_SIZE * 100 / ORIGINAL_SIZE)))
        
        log_success "Size reduction for $MODULE: ${REDUCTION}% (${ORIGINAL_SIZE} → ${OPTIMIZED_SIZE} bytes)"
        
        # Clean up
        rm "${WASM_FILE}.original"
    fi
}

# Merge modules into unified build
merge_modules() {
    log_info "Merging modules into unified build..."
    
    # Create final package directory
    mkdir -p "$UNIFIED_CRATE_DIR/pkg"
    
    # Copy main module files
    cp -r "$UNIFIED_CRATE_DIR/pkg-core/"* "$UNIFIED_CRATE_DIR/pkg/" 2>/dev/null || true
    
    # Merge other modules
    for module in neural forecasting; do
        if [ -d "$UNIFIED_CRATE_DIR/pkg-$module" ]; then
            # Copy WASM files with module prefix
            cp "$UNIFIED_CRATE_DIR/pkg-$module/"*.wasm "$UNIFIED_CRATE_DIR/pkg/" 2>/dev/null || true
            
            # Merge TypeScript definitions
            if [ -f "$UNIFIED_CRATE_DIR/pkg-$module/ruv_swarm_${module}.d.ts" ]; then
                cat "$UNIFIED_CRATE_DIR/pkg-$module/ruv_swarm_${module}.d.ts" >> "$UNIFIED_CRATE_DIR/pkg/ruv_swarm_unified.d.ts"
            fi
        fi
    done
    
    # Generate unified loader
    generate_unified_loader
    
    log_success "Modules merged successfully"
}

# Generate unified JavaScript loader
generate_unified_loader() {
    log_info "Generating unified loader..."
    
    cat > "$UNIFIED_CRATE_DIR/pkg/ruv_swarm_unified_loader.js" << 'EOF'
// Unified loader for ruv-swarm WASM modules

export class RuvSwarmUnified {
    constructor() {
        this.modules = {
            core: null,
            neural: null,
            forecasting: null
        };
        this.initialized = false;
    }
    
    async init(config = {}) {
        if (this.initialized) return;
        
        // Load core module (always required)
        const coreModule = await import('./ruv_swarm_core.js');
        await coreModule.default();
        this.modules.core = coreModule;
        
        // Load optional modules based on config
        if (config.neural !== false) {
            try {
                const neuralModule = await import('./ruv_swarm_neural.js');
                await neuralModule.default();
                this.modules.neural = neuralModule;
            } catch (e) {
                console.warn('Neural module not available:', e);
            }
        }
        
        if (config.forecasting !== false) {
            try {
                const forecastingModule = await import('./ruv_swarm_forecasting.js');
                await forecastingModule.default();
                this.modules.forecasting = forecastingModule;
            } catch (e) {
                console.warn('Forecasting module not available:', e);
            }
        }
        
        this.initialized = true;
        console.log('RuvSwarm WASM modules loaded:', Object.keys(this.modules).filter(k => this.modules[k]));
    }
    
    get core() { return this.modules.core; }
    get neural() { return this.modules.neural; }
    get forecasting() { return this.modules.forecasting; }
}

// Export singleton instance
export const ruvSwarm = new RuvSwarmUnified();
EOF
}

# Generate build report
generate_build_report() {
    log_info "Generating build report..."
    
    local REPORT_FILE="$UNIFIED_CRATE_DIR/pkg/build-report.json"
    local TOTAL_SIZE=0
    local MODULES_INFO="["
    
    for wasm_file in "$UNIFIED_CRATE_DIR/pkg/"*.wasm; do
        if [ -f "$wasm_file" ]; then
            local SIZE=$(stat -c%s "$wasm_file" 2>/dev/null || stat -f%z "$wasm_file")
            local NAME=$(basename "$wasm_file")
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
            
            MODULES_INFO+="{\"name\":\"$NAME\",\"size\":$SIZE},"
        fi
    done
    
    MODULES_INFO="${MODULES_INFO%,}]"
    
    cat > "$REPORT_FILE" << EOF
{
    "buildTime": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "totalSize": $TOTAL_SIZE,
    "totalSizeMB": $(echo "scale=2; $TOTAL_SIZE / 1024 / 1024" | bc),
    "modules": $MODULES_INFO,
    "features": {
        "simd": true,
        "parallel": true,
        "optimize": true
    },
    "rustVersion": "$(rustc --version | cut -d' ' -f2)",
    "wasmPackVersion": "$(wasm-pack --version | cut -d' ' -f2)"
}
EOF
    
    log_success "Build report generated: $REPORT_FILE"
}

# Run tests
run_tests() {
    log_info "Running WASM tests..."
    
    cd "$UNIFIED_CRATE_DIR"
    
    # Run wasm-pack tests
    wasm-pack test --headless --chrome
    
    log_success "Tests completed"
}

# Main build pipeline
main() {
    local BUILD_ALL=true
    local RUN_TESTS=false
    local MODULES_TO_BUILD=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --module)
                BUILD_ALL=false
                MODULES_TO_BUILD+=("$2")
                shift 2
                ;;
            --test)
                RUN_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --module MODULE    Build specific module (core, neural, forecasting)"
                echo "  --test            Run tests after build"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Start build process
    log_info "Starting WASM build orchestration..."
    
    # Initialize environment
    init_build_env
    
    # Determine modules to build
    if [ "$BUILD_ALL" = true ]; then
        MODULES_TO_BUILD=("core" "neural" "forecasting")
    fi
    
    # Build modules
    for module in "${MODULES_TO_BUILD[@]}"; do
        build_module "$module"
    done
    
    # Merge if building all
    if [ "$BUILD_ALL" = true ]; then
        merge_modules
    fi
    
    # Run tests if requested
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi
    
    # Generate report
    generate_build_report
    
    # Final summary
    log_success "WASM build orchestration completed!"
    
    # Display summary
    if [ -f "$UNIFIED_CRATE_DIR/pkg/build-report.json" ]; then
        local TOTAL_SIZE_MB=$(jq -r '.totalSizeMB' "$UNIFIED_CRATE_DIR/pkg/build-report.json")
        log_info "Total WASM size: ${TOTAL_SIZE_MB}MB"
        
        if (( $(echo "$TOTAL_SIZE_MB > 5" | bc -l) )); then
            log_warning "Total size exceeds 5MB target!"
        else
            log_success "Size target achieved!"
        fi
    fi
}

# Run main
main "$@"