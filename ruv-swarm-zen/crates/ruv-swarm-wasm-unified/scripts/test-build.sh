#!/bin/bash
# Test script to validate WASM build system

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UNIFIED_CRATE_DIR="$SCRIPT_DIR/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local TEST_NAME=$1
    local TEST_CMD=$2
    
    echo -e "${BLUE}Running test:${NC} $TEST_NAME"
    
    if eval "$TEST_CMD"; then
        echo -e "${GREEN}✓ PASSED:${NC} $TEST_NAME"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED:${NC} $TEST_NAME"
        ((TESTS_FAILED++))
    fi
}

# Test 1: Check Rust installation
run_test "Rust installation" "command -v rustc &> /dev/null"

# Test 2: Check wasm32 target
run_test "WASM target installed" "rustup target list --installed | grep -q wasm32-unknown-unknown"

# Test 3: Check wasm-pack
run_test "wasm-pack available" "command -v wasm-pack &> /dev/null || (echo 'Installing wasm-pack...' && curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh)"

# Test 4: Validate Cargo.toml
run_test "Cargo.toml valid" "cd $UNIFIED_CRATE_DIR && cargo check --no-default-features"

# Test 5: Build core module
run_test "Build core module" "cd $UNIFIED_CRATE_DIR && cargo build --release --target wasm32-unknown-unknown"

# Test 6: Run Rust tests
run_test "Rust unit tests" "cd $UNIFIED_CRATE_DIR && cargo test --lib"

# Test 7: Build with wasm-pack
run_test "wasm-pack build" "cd $UNIFIED_CRATE_DIR && wasm-pack build --target web --dev"

# Test 8: Check generated files
run_test "WASM file generated" "[ -f $UNIFIED_CRATE_DIR/pkg/ruv_swarm_wasm_unified_bg.wasm ]"
run_test "JS bindings generated" "[ -f $UNIFIED_CRATE_DIR/pkg/ruv_swarm_wasm_unified.js ]"
run_test "TypeScript definitions generated" "[ -f $UNIFIED_CRATE_DIR/pkg/ruv_swarm_wasm_unified.d.ts ]"

# Test 9: Check WASM size
if [ -f "$UNIFIED_CRATE_DIR/pkg/ruv_swarm_wasm_unified_bg.wasm" ]; then
    WASM_SIZE=$(stat -c%s "$UNIFIED_CRATE_DIR/pkg/ruv_swarm_wasm_unified_bg.wasm" 2>/dev/null || stat -f%z "$UNIFIED_CRATE_DIR/pkg/ruv_swarm_wasm_unified_bg.wasm")
    WASM_SIZE_MB=$((WASM_SIZE / 1024 / 1024))
    
    if [ $WASM_SIZE_MB -lt 5 ]; then
        echo -e "${GREEN}✓ PASSED:${NC} WASM size check (${WASM_SIZE_MB}MB < 5MB)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED:${NC} WASM size check (${WASM_SIZE_MB}MB >= 5MB)"
        ((TESTS_FAILED++))
    fi
fi

# Test 10: Validate package.json
run_test "package.json valid" "[ -f $UNIFIED_CRATE_DIR/pkg/package.json ] && jq . $UNIFIED_CRATE_DIR/pkg/package.json > /dev/null"

# Summary
echo -e "\n${BLUE}Test Summary:${NC}"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC} ✨"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC} Please check the output above."
    exit 1
fi