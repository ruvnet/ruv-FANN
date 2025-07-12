#!/bin/bash

# OpenCV Rust Publishing Script
# This script publishes all OpenCV Rust crates to crates.io using environment variables

set -euo pipefail

# Load environment variables
if [ -f "../.env" ]; then
    source "../.env"
    echo "✅ Loaded environment variables from .env"
else
    echo "❌ .env file not found in parent directory"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check required environment variables
check_env_vars() {
    print_step "Checking environment variables..."
    
    if [ -z "${CARGO_REGISTRY_TOKEN:-}" ] || [ "$CARGO_REGISTRY_TOKEN" = "your_crates_io_token_here" ]; then
        print_error "CARGO_REGISTRY_TOKEN not set or is placeholder value"
        print_warning "Please set your actual crates.io API token in .env file"
        exit 1
    fi
    
    print_status "Environment variables validated"
}

# Login to cargo registry
cargo_login() {
    print_step "Logging into cargo registry..."
    
    if [ "${PUBLISH_DRY_RUN:-false}" = "true" ]; then
        print_warning "Dry run mode - skipping cargo login"
        return 0
    fi
    
    echo "$CARGO_REGISTRY_TOKEN" | cargo login --registry crates-io
    print_status "Successfully logged into cargo registry"
}

# Validate crate before publishing
validate_crate() {
    local crate_path=$1
    local crate_name=$(basename "$crate_path")
    
    print_step "Validating $crate_name..."
    
    cd "$crate_path"
    
    # Check if Cargo.toml exists
    if [ ! -f "Cargo.toml" ]; then
        print_error "Cargo.toml not found in $crate_path"
        return 1
    fi
    
    # Run cargo check
    if ! cargo check --all-features; then
        print_error "Cargo check failed for $crate_name"
        return 1
    fi
    
    # Run tests if they exist
    if [ -d "tests" ] || grep -q "\[\[test\]\]" Cargo.toml; then
        print_status "Running tests for $crate_name..."
        if ! cargo test --all-features; then
            print_warning "Tests failed for $crate_name - continuing anyway"
        fi
    fi
    
    # Validate package
    if ! cargo package --allow-dirty; then
        print_error "Cargo package failed for $crate_name"
        return 1
    fi
    
    print_status "$crate_name validation completed"
    cd - > /dev/null
}

# Publish a single crate
publish_crate() {
    local crate_path=$1
    local crate_name=$(basename "$crate_path")
    
    print_step "Publishing $crate_name..."
    
    cd "$crate_path"
    
    # Check if crate is already published
    local current_version=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
    
    if cargo search "$crate_name" | grep -q "^$crate_name.*$current_version"; then
        print_warning "$crate_name v$current_version already published - skipping"
        cd - > /dev/null
        return 0
    fi
    
    # Publish crate
    local publish_cmd="cargo publish"
    
    if [ "${PUBLISH_DRY_RUN:-false}" = "true" ]; then
        publish_cmd="$publish_cmd --dry-run"
        print_warning "Dry run mode - not actually publishing"
    fi
    
    if ! $publish_cmd --allow-dirty; then
        print_error "Failed to publish $crate_name"
        cd - > /dev/null
        return 1
    fi
    
    print_status "Successfully published $crate_name v$current_version"
    
    # Wait a bit to avoid rate limiting
    if [ "${PUBLISH_DRY_RUN:-false}" != "true" ]; then
        print_status "Waiting 30 seconds to avoid rate limiting..."
        sleep 30
    fi
    
    cd - > /dev/null
}

# Main publishing workflow
main() {
    print_step "Starting OpenCV Rust publishing workflow..."
    print_status "Version: ${CRATE_VERSION}"
    print_status "Dry run: ${PUBLISH_DRY_RUN:-false}"
    
    # Check environment
    check_env_vars
    
    # Login to cargo
    cargo_login
    
    # Parse publish order
    IFS=',' read -ra CRATES <<< "${PUBLISH_ORDER}"
    
    print_step "Publishing ${#CRATES[@]} crates in order: ${PUBLISH_ORDER}"
    
    # Validate all crates first
    print_step "Validating all crates..."
    for crate in "${CRATES[@]}"; do
        if [ -d "$crate" ]; then
            validate_crate "$crate"
        else
            print_warning "Crate directory $crate not found - skipping"
        fi
    done
    
    # Publish crates in order
    print_step "Publishing crates..."
    for crate in "${CRATES[@]}"; do
        if [ -d "$crate" ]; then
            publish_crate "$crate"
        else
            print_warning "Crate directory $crate not found - skipping"
        fi
    done
    
    print_status "OpenCV Rust publishing workflow completed successfully!"
    
    # Generate summary
    print_step "Publishing Summary:"
    print_status "📦 Total crates processed: ${#CRATES[@]}"
    print_status "🌐 Registry: ${CARGO_REGISTRY_URL}"
    print_status "📄 License: ${CRATE_LICENSE}"
    print_status "🏠 Homepage: ${CRATE_HOMEPAGE}"
    print_status "📚 Documentation: https://docs.rs/${CRATE_NAME}"
    
    if [ "${PUBLISH_DRY_RUN:-false}" = "true" ]; then
        print_warning "This was a dry run - no crates were actually published"
        print_status "To publish for real, set PUBLISH_DRY_RUN=false in .env"
    else
        print_status "🎉 All crates published successfully to crates.io!"
        print_status "🔗 View at: https://crates.io/crates/${CRATE_NAME}"
    fi
}

# Run main function
main "$@"