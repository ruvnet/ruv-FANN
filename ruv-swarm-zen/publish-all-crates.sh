#!/bin/bash

# RUV-Swarm Crates Publishing Script
# Publishes all ruv-swarm crates to crates.io in proper dependency order

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=${DRY_RUN:-false}
FORCE_PUBLISH=${FORCE_PUBLISH:-false}
SKIP_TESTS=${SKIP_TESTS:-false}

echo -e "${BLUE}🚀 RUV-Swarm Crates Publishing Script${NC}"
echo "=================================="

# Check if token is available
if [ -z "$CARGO_REGISTRY_TOKEN" ]; then
    if [ -f ".env" ]; then
        echo -e "${YELLOW}📝 Loading token from .env file...${NC}"
        source .env
    fi
    
    if [ -z "$CARGO_REGISTRY_TOKEN" ]; then
        echo -e "${RED}❌ Error: CARGO_REGISTRY_TOKEN not set${NC}"
        echo "Please set your crates.io token:"
        echo "  export CARGO_REGISTRY_TOKEN=your_token_here"
        echo "Or add it to .env file"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Cargo registry token found${NC}"

# Function to check if crate is already published
check_crate_published() {
    local crate_name=$1
    local version=$2
    
    echo -e "${BLUE}🔍 Checking if $crate_name v$version is already published...${NC}"
    
    # Query crates.io API
    local response=$(curl -s "https://crates.io/api/v1/crates/$crate_name" || echo "not_found")
    
    if echo "$response" | grep -q "\"num\":\"$version\""; then
        echo -e "${YELLOW}⚠️  $crate_name v$version already published, skipping${NC}"
        return 0
    else
        echo -e "${GREEN}📦 $crate_name v$version not published, will publish${NC}"
        return 1
    fi
}

# Function to extract version from Cargo.toml
get_crate_version() {
    local cargo_toml=$1
    grep '^version = ' "$cargo_toml" | head -1 | sed 's/version = "\(.*\)"/\1/'
}

# Function to publish a single crate
publish_crate() {
    local crate_path=$1
    local crate_name=$(basename "$crate_path")
    
    echo -e "\n${BLUE}📦 Processing crate: $crate_name${NC}"
    echo "Path: $crate_path"
    
    # Check if Cargo.toml exists
    if [ ! -f "$crate_path/Cargo.toml" ]; then
        echo -e "${RED}❌ No Cargo.toml found in $crate_path${NC}"
        return 1
    fi
    
    # Get version
    local version=$(get_crate_version "$crate_path/Cargo.toml")
    echo "Version: $version"
    
    # Check if already published (unless force)
    if [ "$FORCE_PUBLISH" != "true" ]; then
        if check_crate_published "$crate_name" "$version"; then
            return 0
        fi
    fi
    
    cd "$crate_path"
    
    # Temporarily modify Cargo.toml to use only path dependencies for publishing
    if [ -f "Cargo.toml.backup" ]; then
        echo -e "${YELLOW}⚠️  Backup already exists, restoring first${NC}"
        mv Cargo.toml.backup Cargo.toml
    fi
    
    # Create backup and modify for publishing
    cp Cargo.toml Cargo.toml.backup
    
    # Remove version requirements from path dependencies for publishing
    sed -i 's/ruv-swarm-\([a-z-]*\) = { path = "\([^"]*\)", version = "[^"]*"/ruv-swarm-\1 = { path = "\2"/g' Cargo.toml
    
    # Run tests unless skipped
    if [ "$SKIP_TESTS" != "true" ]; then
        echo -e "${BLUE}🧪 Running tests for $crate_name...${NC}"
        if ! cargo test --release; then
            echo -e "${RED}❌ Tests failed for $crate_name${NC}"
            # Restore original Cargo.toml
            mv Cargo.toml.backup Cargo.toml
            return 1
        fi
        echo -e "${GREEN}✅ Tests passed for $crate_name${NC}"
    fi
    
    # Check if this is a dry run
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}🔍 DRY RUN: Would publish $crate_name v$version${NC}"
        cargo publish --dry-run --allow-dirty
        
        # Restore original Cargo.toml
        mv Cargo.toml.backup Cargo.toml
        return 0
    fi
    
    # Publish the crate
    echo -e "${BLUE}🚀 Publishing $crate_name v$version...${NC}"
    if cargo publish --token "$CARGO_REGISTRY_TOKEN" --allow-dirty; then
        echo -e "${GREEN}✅ Successfully published $crate_name v$version${NC}"
        
        # Restore original Cargo.toml
        mv Cargo.toml.backup Cargo.toml
        
        # Wait a bit for crates.io to process
        echo -e "${YELLOW}⏳ Waiting 30 seconds for crates.io to process...${NC}"
        sleep 30
        
        return 0
    else
        echo -e "${RED}❌ Failed to publish $crate_name${NC}"
        # Restore original Cargo.toml
        mv Cargo.toml.backup Cargo.toml
        return 1
    fi
}

# Main script
main() {
    local script_dir=$(dirname "$0")
    cd "$script_dir"
    
    echo -e "\n${BLUE}🎯 Starting crate publishing process${NC}"
    
    # Define publishing order (dependencies first)
    local publish_order=(
        # Core crates (no dependencies)
        "crates/ruv-swarm-core"
        "crates/claude-parser"
        
        # Transport and persistence (minimal dependencies)
        "crates/ruv-swarm-transport"
        "crates/ruv-swarm-persistence"
        
        # Agents (depends on core)
        "crates/ruv-swarm-agents"
        
        # ML components (depends on core)
        "crates/ruv-swarm-ml"
        
        # WASM (depends on core, agents)
        "crates/ruv-swarm-wasm"
        
        # MCP (depends on core, agents, transport)
        "crates/ruv-swarm-mcp"
        
        # Adapters (depends on core, agents, persistence)
        "crates/swe-bench-adapter"
        
        # CLI (depends on everything)
        "crates/ruv-swarm-cli"
        
        # Training and benchmarking
        "ml-training"
        "benchmarking"
    )
    
    local published_count=0
    local skipped_count=0
    local failed_count=0
    
    # Process each crate in order
    for crate_path in "${publish_order[@]}"; do
        if [ -d "$crate_path" ]; then
            if publish_crate "$crate_path"; then
                if [ "$DRY_RUN" = "true" ]; then
                    ((skipped_count++))
                else
                    ((published_count++))
                fi
            else
                ((failed_count++))
                if [ "$FORCE_PUBLISH" != "true" ]; then
                    echo -e "${RED}❌ Publishing failed, stopping. Use FORCE_PUBLISH=true to continue.${NC}"
                    break
                fi
            fi
        else
            echo -e "${YELLOW}⚠️  Crate path $crate_path not found, skipping${NC}"
            ((skipped_count++))
        fi
    done
    
    # Summary
    echo -e "\n${BLUE}📊 Publishing Summary${NC}"
    echo "==================="
    
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}🔍 DRY RUN COMPLETED${NC}"
        echo "Crates that would be published: $skipped_count"
    else
        echo -e "${GREEN}✅ Successfully published: $published_count${NC}"
    fi
    
    echo -e "${YELLOW}⏭️  Skipped: $skipped_count${NC}"
    echo -e "${RED}❌ Failed: $failed_count${NC}"
    
    if [ $failed_count -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All crates processed successfully!${NC}"
        
        if [ "$DRY_RUN" != "true" ]; then
            echo -e "\n${BLUE}📝 Next steps:${NC}"
            echo "1. Check crates.io for your published crates"
            echo "2. Update documentation with new version numbers"
            echo "3. Tag the release in git"
            echo "4. Announce the release"
        fi
    else
        echo -e "\n${RED}⚠️  Some crates failed to publish. Check the output above.${NC}"
        exit 1
    fi
}

# Help function
show_help() {
    echo "RUV-Swarm Crates Publishing Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run         Perform a dry run (don't actually publish)"
    echo "  --force           Force publish even if version exists"
    echo "  --skip-tests      Skip running tests before publishing"
    echo "  --help            Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  CARGO_REGISTRY_TOKEN  Your crates.io API token (required)"
    echo "  DRY_RUN              Set to 'true' for dry run"
    echo "  FORCE_PUBLISH        Set to 'true' to force publish"
    echo "  SKIP_TESTS           Set to 'true' to skip tests"
    echo ""
    echo "Examples:"
    echo "  $0                    # Normal publish"
    echo "  $0 --dry-run         # Dry run"
    echo "  DRY_RUN=true $0      # Dry run via env var"
    echo "  $0 --force           # Force publish all crates"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE_PUBLISH=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main