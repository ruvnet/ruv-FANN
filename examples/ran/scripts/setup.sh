#!/bin/bash

# RAN Intelligence Platform Setup Script
# This script sets up the development environment and builds all services

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    log_info "Checking requirements..."
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        log_error "Rust/Cargo not found. Please install Rust from https://rustup.rs/"
        exit 1
    fi
    log_success "Rust/Cargo found: $(cargo --version)"
    
    # Check protoc
    if ! command -v protoc &> /dev/null; then
        log_error "Protocol Buffer compiler not found. Please install protobuf-compiler"
        log_info "On Ubuntu/Debian: sudo apt-get install protobuf-compiler"
        log_info "On macOS: brew install protobuf"
        exit 1
    fi
    log_success "protoc found: $(protoc --version)"
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        log_success "Docker found: $(docker --version)"
    else
        log_warning "Docker not found. Docker is optional but recommended for deployment."
    fi
    
    # Check PostgreSQL client (optional)
    if command -v psql &> /dev/null; then
        log_success "PostgreSQL client found: $(psql --version)"
    else
        log_warning "PostgreSQL client not found. This is optional for development."
    fi
}

# Install Rust components
install_rust_components() {
    log_info "Installing Rust components..."
    
    # Install required components
    rustup component add rustfmt clippy
    
    # Install useful cargo tools
    if ! command -v cargo-audit &> /dev/null; then
        log_info "Installing cargo-audit..."
        cargo install cargo-audit
    fi
    
    if ! command -v cargo-watch &> /dev/null; then
        log_info "Installing cargo-watch..."
        cargo install cargo-watch
    fi
    
    log_success "Rust components installed"
}

# Build all services
build_services() {
    log_info "Building all services..."
    
    # Build in release mode for better performance
    cargo build --release --all-features
    
    if [ $? -eq 0 ]; then
        log_success "All services built successfully"
    else
        log_error "Build failed"
        exit 1
    fi
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Run unit tests
    cargo test --all-features
    
    if [ $? -eq 0 ]; then
        log_success "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

# Format code
format_code() {
    log_info "Formatting code..."
    cargo fmt --all
    log_success "Code formatted"
}

# Run linting
run_linting() {
    log_info "Running linting..."
    cargo clippy --all-targets --all-features -- -D warnings
    
    if [ $? -eq 0 ]; then
        log_success "Linting passed"
    else
        log_error "Linting failed"
        exit 1
    fi
}

# Setup database (if PostgreSQL is available)
setup_database() {
    if command -v psql &> /dev/null; then
        log_info "Setting up database..."
        
        # Check if database exists
        if psql -lqt | cut -d \| -f 1 | grep -qw ran_intelligence; then
            log_success "Database 'ran_intelligence' already exists"
        else
            log_info "Creating database 'ran_intelligence'..."
            createdb ran_intelligence || log_warning "Failed to create database. You may need to create it manually."
        fi
    else
        log_warning "PostgreSQL not available. Database setup skipped."
    fi
}

# Generate documentation
generate_docs() {
    log_info "Generating documentation..."
    cargo doc --all-features --no-deps --open
    log_success "Documentation generated"
}

# Setup development environment
setup_dev_env() {
    log_info "Setting up development environment..."
    
    # Create necessary directories
    mkdir -p data models config logs
    
    # Create sample configuration files
    if [ ! -f "config/development.toml" ]; then
        cat > config/development.toml << EOF
[server]
host = "127.0.0.1"
port = 50051
metrics_port = 9090

[database]
url = "postgresql://localhost:5432/ran_intelligence"
max_connections = 10

[logging]
level = "info"
format = "json"

[data]
input_path = "./data/input"
output_path = "./data/output"
models_path = "./models"

[ml]
batch_size = 32
learning_rate = 0.001
epochs = 100
validation_split = 0.2
EOF
        log_success "Created development configuration"
    fi
    
    log_success "Development environment setup complete"
}

# Main function
main() {
    log_info "RAN Intelligence Platform Setup"
    log_info "=============================="
    
    # Parse command line arguments
    SKIP_TESTS=false
    SKIP_BUILD=false
    GENERATE_DOCS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --docs)
                GENERATE_DOCS=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-tests    Skip running tests"
                echo "  --skip-build    Skip building services"
                echo "  --docs          Generate and open documentation"
                echo "  --help, -h      Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_requirements
    install_rust_components
    setup_dev_env
    format_code
    
    if [ "$SKIP_BUILD" = false ]; then
        build_services
    fi
    
    run_linting
    
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    fi
    
    setup_database
    
    if [ "$GENERATE_DOCS" = true ]; then
        generate_docs
    fi
    
    log_success "Setup complete!"
    log_info "Next steps:"
    log_info "1. Review the configuration in config/development.toml"
    log_info "2. Start the services using: ./scripts/start-services.sh"
    log_info "3. Or use Docker: docker-compose up"
    log_info "4. View the documentation at: file://$(pwd)/target/doc/ran_intelligence_platform/index.html"
}

# Run main function
main "$@"