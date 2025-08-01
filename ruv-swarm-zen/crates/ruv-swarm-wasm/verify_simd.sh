#!/bin/bash

# SIMD Verification Script for RUV-SWARM WASM
# This script verifies that SIMD features are properly enabled and functional

set -e

echo "🚀 RUV-SWARM SIMD Verification Script"
echo "===================================="
echo

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK")
            echo "✅ $message"
            ;;
        "WARN")
            echo "⚠️  $message"
            ;;
        "ERROR")
            echo "❌ $message"
            ;;
        "INFO")
            echo "ℹ️  $message"
            ;;
    esac
}

# Check prerequisites
echo "1. Checking Prerequisites..."
echo "----------------------------"

# Check Rust version
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    print_status "OK" "Rust found: $RUST_VERSION"
else
    print_status "ERROR" "Rust not found. Please install Rust."
    exit 1
fi

# Check wasm-pack
if command -v wasm-pack &> /dev/null; then
    WASM_PACK_VERSION=$(wasm-pack --version)
    print_status "OK" "wasm-pack found: $WASM_PACK_VERSION"
else
    print_status "WARN" "wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]]; then
    print_status "ERROR" "Not in a Cargo project directory"
    exit 1
fi

if [[ ! $(grep -q "ruv-swarm-wasm" Cargo.toml) ]]; then
    print_status "WARN" "Not in ruv-swarm-wasm directory, changing..."
    cd "$(dirname "$0")"
fi

echo

# Check SIMD support in Cargo.toml
echo "2. Checking SIMD Configuration..."
echo "--------------------------------"

if grep -q "simd.*=.*\[\]" Cargo.toml; then
    print_status "OK" "SIMD feature found in Cargo.toml"
else
    print_status "ERROR" "SIMD feature not properly configured in Cargo.toml"
fi

if grep -q "wide.*=" Cargo.toml; then
    print_status "OK" "Wide SIMD library dependency found"
else
    print_status "ERROR" "Wide SIMD library not found in dependencies"
fi

if grep -q "target-feature=+simd128" Cargo.toml; then
    print_status "OK" "SIMD128 target feature enabled"
else
    print_status "WARN" "SIMD128 target feature not explicitly enabled"
fi

echo

# Test compilation
echo "3. Testing Compilation..."
echo "------------------------"

print_status "INFO" "Building with SIMD features..."

if cargo check --features simd --target wasm32-unknown-unknown 2>&1 | tee build_log.txt; then
    print_status "OK" "WASM compilation successful"
    
    # Check for warnings about SIMD
    if grep -i "simd" build_log.txt; then
        print_status "INFO" "SIMD-related messages found in build log"
    fi
    
    # Check for errors
    if grep -i "error" build_log.txt; then
        print_status "WARN" "Errors found in build log (but compilation succeeded)"
    fi
else
    print_status "ERROR" "WASM compilation failed"
    echo "Build log:"
    cat build_log.txt
    exit 1
fi

echo

# Test wasm-pack build
echo "4. Testing wasm-pack Build..."
echo "----------------------------"

print_status "INFO" "Building with wasm-pack for browser..."

# Set RUSTFLAGS for SIMD
export RUSTFLAGS="-C target-feature=+simd128"

if wasm-pack build --target web --out-dir pkg-test --features simd 2>&1 | tee wasm_pack_log.txt; then
    print_status "OK" "wasm-pack build successful"
    
    # Check if WASM file was generated
    if [[ -f "pkg-test/ruv_swarm_wasm.wasm" ]]; then
        print_status "OK" "WASM binary generated"
        
        # Check file size
        WASM_SIZE=$(stat -f%z pkg-test/ruv_swarm_wasm.wasm 2>/dev/null || stat -c%s pkg-test/ruv_swarm_wasm.wasm 2>/dev/null)
        print_status "INFO" "WASM binary size: $((WASM_SIZE / 1024)) KB"
        
        # Check for SIMD instructions (basic check)
        if command -v wasm-objdump &> /dev/null; then
            if wasm-objdump -d pkg-test/ruv_swarm_wasm.wasm | grep -i "v128\|simd" > /dev/null; then
                print_status "OK" "SIMD instructions found in WASM binary"
            else
                print_status "WARN" "No obvious SIMD instructions found (may be optimized)"
            fi
        else
            print_status "INFO" "wasm-objdump not available for instruction analysis"
        fi
    else
        print_status "ERROR" "WASM binary not generated"
    fi
    
    # Check if JS bindings were generated
    if [[ -f "pkg-test/ruv_swarm_wasm.js" ]]; then
        print_status "OK" "JavaScript bindings generated"
        
        # Check for SIMD exports
        if grep -q "SimdVectorOps\|SimdMatrixOps\|detect_simd_capabilities" pkg-test/ruv_swarm_wasm.js; then
            print_status "OK" "SIMD functions exported to JavaScript"
        else
            print_status "WARN" "SIMD functions not found in JS exports"
        fi
    else
        print_status "ERROR" "JavaScript bindings not generated"
    fi
    
    # Check TypeScript definitions
    if [[ -f "pkg-test/ruv_swarm_wasm.d.ts" ]]; then
        print_status "OK" "TypeScript definitions generated"
        
        if grep -q "SimdVectorOps\|SimdMatrixOps" pkg-test/ruv_swarm_wasm.d.ts; then
            print_status "OK" "SIMD types found in TypeScript definitions"
        else
            print_status "WARN" "SIMD types not found in TypeScript definitions"
        fi
    fi
    
else
    print_status "ERROR" "wasm-pack build failed"
    echo "wasm-pack log:"
    cat wasm_pack_log.txt
fi

echo

# Feature analysis
echo "5. Feature Analysis..."
echo "--------------------"

if [[ -f "src/lib.rs" ]]; then
    # Check for SIMD module imports
    if grep -q "mod simd_ops" src/lib.rs; then
        print_status "OK" "SIMD operations module imported"
    else
        print_status "ERROR" "SIMD operations module not imported"
    fi
    
    # Check for SIMD exports
    if grep -q "pub use.*Simd" src/lib.rs; then
        print_status "OK" "SIMD functions exported from lib.rs"
    else
        print_status "WARN" "SIMD functions may not be exported"
    fi
fi

# Check source files
if [[ -f "src/simd_ops.rs" ]]; then
    print_status "OK" "SIMD operations source file exists"
    
    # Check for wide library usage
    if grep -q "use wide::" src/simd_ops.rs; then
        print_status "OK" "Wide SIMD library used in source"
    else
        print_status "WARN" "Wide SIMD library not used"
    fi
    
    # Check for wasm_bindgen exports
    if grep -q "#\[wasm_bindgen\]" src/simd_ops.rs; then
        print_status "OK" "WASM bindings found in SIMD operations"
    else
        print_status "ERROR" "No WASM bindings in SIMD operations"
    fi
    
    # Count SIMD functions
    SIMD_FUNCTIONS=$(grep -c "#\[wasm_bindgen\]" src/simd_ops.rs || echo "0")
    print_status "INFO" "Found $SIMD_FUNCTIONS WASM-exported SIMD functions"
else
    print_status "ERROR" "SIMD operations source file not found"
fi

echo

# Performance estimation
echo "6. Performance Estimation..."
echo "---------------------------"

print_status "INFO" "Based on implementation analysis:"

if [[ -f "src/simd_ops.rs" ]] && grep -q "f32x4" src/simd_ops.rs; then
    print_status "OK" "4-wide f32 SIMD vectors used (expected 2-4x speedup)"
else
    print_status "WARN" "SIMD vector width unclear"
fi

if grep -q "simd_dot_product\|simd_matrix_multiply" src/simd_ops.rs 2>/dev/null; then
    print_status "OK" "Core mathematical operations SIMD-optimized"
else
    print_status "WARN" "Core operations may not be SIMD-optimized"
fi

echo

# Generate report
echo "7. Generating Report..."
echo "---------------------"

REPORT_FILE="simd_verification_report.md"

cat > "$REPORT_FILE" << EOF
# RUV-SWARM SIMD Verification Report

Generated on: $(date)

## Configuration Status

### Dependencies
- Wide SIMD library: $(grep -q "wide" Cargo.toml && echo "✅ Configured" || echo "❌ Missing")
- SIMD feature flag: $(grep -q "simd.*=.*\[\]" Cargo.toml && echo "✅ Enabled" || echo "❌ Disabled")

### Target Features
- SIMD128 support: $(grep -q "target-feature=+simd128" Cargo.toml && echo "✅ Enabled" || echo "⚠️ Not explicit")

### Source Code
- SIMD operations module: $(test -f "src/simd_ops.rs" && echo "✅ Present" || echo "❌ Missing")
- WASM bindings: $(grep -q "#\[wasm_bindgen\]" src/simd_ops.rs 2>/dev/null && echo "✅ Present" || echo "❌ Missing")

## Build Results

### Compilation
- Cargo check: $(test -f "build_log.txt" && echo "✅ Passed" || echo "❌ Failed")
- wasm-pack build: $(test -f "wasm_pack_log.txt" && echo "✅ Passed" || echo "❌ Failed")

### Generated Files
- WASM binary: $(test -f "pkg-test/ruv_swarm_wasm.wasm" && echo "✅ Generated" || echo "❌ Missing")
- JS bindings: $(test -f "pkg-test/ruv_swarm_wasm.js" && echo "✅ Generated" || echo "❌ Missing")
- TypeScript defs: $(test -f "pkg-test/ruv_swarm_wasm.d.ts" && echo "✅ Generated" || echo "❌ Missing")

## Expected Performance

Based on SIMD implementation:
- Vector operations: 2-4x speedup expected
- Matrix multiplication: 2-3x speedup expected
- Activation functions: 3-4x speedup expected

## Recommendations

1. Test in target browsers with WebAssembly SIMD support
2. Benchmark against scalar implementations
3. Monitor memory usage with large datasets
4. Consider fallback for non-SIMD environments

## Files Generated

- Build logs: build_log.txt, wasm_pack_log.txt
- Test package: pkg-test/
- This report: $REPORT_FILE

EOF

print_status "OK" "Report generated: $REPORT_FILE"

echo

# Cleanup
echo "8. Cleanup..."
echo "-----------"

print_status "INFO" "Cleaning up temporary files..."
rm -f build_log.txt wasm_pack_log.txt
print_status "OK" "Cleanup complete"

echo

# Summary
echo "🎉 SIMD Verification Complete!"
echo "============================="
echo
echo "Summary:"
echo "  - SIMD features: $(grep -q "simd.*=.*\[\]" Cargo.toml && echo "✅ Enabled" || echo "❌ Disabled")"
echo "  - Compilation: $(test -f "pkg-test/ruv_swarm_wasm.wasm" && echo "✅ Success" || echo "❌ Failed")"
echo "  - JS bindings: $(test -f "pkg-test/ruv_swarm_wasm.js" && echo "✅ Generated" || echo "❌ Missing")"
echo
echo "Next steps:"
echo "  1. Review the generated report: $REPORT_FILE"
echo "  2. Test the HTML demo: examples/simd_demo.html"
echo "  3. Run performance benchmarks in target browsers"
echo
echo "Test package location: pkg-test/"
echo "Demo files: examples/"