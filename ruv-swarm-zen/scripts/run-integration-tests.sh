#!/bin/bash
# Script to run comprehensive integration tests for ruv-swarm

set -e

echo "=== Running ruv-swarm Integration Tests ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print test section
print_section() {
    echo -e "${YELLOW}>>> $1${NC}"
    echo
}

# Function to run test with output
run_test() {
    local test_name=$1
    echo -e "Running: ${GREEN}$test_name${NC}"
    
    if cargo test --test integration_test "$test_name" -- --nocapture; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        exit 1
    fi
    echo
}

# Change to project root
cd "$(dirname "$0")/.."

print_section "Building project"
cargo build --all-features

print_section "Running integration tests"

# Run individual integration tests
run_test "test_end_to_end_claude_code_swe_bench"
run_test "test_stream_parsing_and_metrics_collection"
run_test "test_model_training_and_evaluation"
run_test "test_performance_improvement_validation"
run_test "test_fibonacci_example"
run_test "test_swarm_performance_scaling"
run_test "test_persistence_and_recovery"
run_test "test_error_handling_and_retry"

print_section "Running all integration tests together"
cargo test --test integration_test -- --test-threads=1

print_section "Running fibonacci demo"
cargo run --example fibonacci_demo

print_section "Running benchmarks (if available)"
if command -v cargo-criterion &> /dev/null; then
    cargo criterion --test integration_test
else
    echo "cargo-criterion not installed, skipping benchmarks"
    echo "Install with: cargo install cargo-criterion"
fi

echo
echo -e "${GREEN}=== All integration tests passed! ===${NC}"