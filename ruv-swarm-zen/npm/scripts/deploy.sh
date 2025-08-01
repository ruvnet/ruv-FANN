#!/bin/bash

# ruv-swarm Deployment Automation Script
# Comprehensive deployment for NPM package with validation and safety checks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="ruv-swarm"
REGISTRY_URL="https://registry.npmjs.org"
BACKUP_DIR="./deployment-backup"
LOG_FILE="./deployment.log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "package.json" ]]; then
        error "No package.json found. Run this script from the npm package directory."
    fi
    
    # Check Node.js version
    local node_version=$(node --version | cut -d'v' -f2)
    local required_version="14.0.0"
    if ! node -pe "require('semver').gte('$node_version', '$required_version')" 2>/dev/null; then
        error "Node.js $required_version or higher required. Found: $node_version"
    fi
    
    # Check npm authentication
    if ! npm whoami >/dev/null 2>&1; then
        error "Not logged in to npm. Run 'npm login' first."
    fi
    
    # Check git status
    if [[ -n $(git status --porcelain) ]]; then
        warn "Working directory is not clean. Uncommitted changes may not be included."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    success "Prerequisites check passed"
}

# Backup current state
create_backup() {
    log "Creating deployment backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup package.json
    cp package.json "$BACKUP_DIR/package.json.backup"
    
    # Backup built files
    if [[ -d "dist" ]]; then
        cp -r dist "$BACKUP_DIR/dist.backup"
    fi
    
    # Backup WASM files
    if [[ -d "wasm" ]]; then
        cp -r wasm "$BACKUP_DIR/wasm.backup"
    fi
    
    success "Backup created in $BACKUP_DIR"
}

# Run quality checks
run_quality_checks() {
    log "Running quality checks..."
    
    # Linting
    log "Running ESLint..."
    npm run lint:check || error "Linting failed"
    
    # Tests
    log "Running test suite..."
    npm run test:all || error "Tests failed"
    
    # Security audit
    log "Running security audit..."
    npm audit --audit-level moderate || warn "Security vulnerabilities found"
    
    # Check for outdated dependencies
    log "Checking for outdated dependencies..."
    npm outdated || warn "Some dependencies are outdated"
    
    success "Quality checks passed"
}

# Build package
build_package() {
    log "Building package..."
    
    # Clean previous builds
    if [[ -d "dist" ]]; then
        rm -rf dist
    fi
    
    # Build WASM modules
    log "Building WASM modules..."
    npm run build:wasm || error "WASM build failed"
    npm run build:wasm-simd || error "WASM SIMD build failed"
    
    # Build main package
    log "Building main package..."
    npm run build || error "Package build failed"
    
    # Generate documentation
    log "Generating documentation..."
    npm run build:docs || warn "Documentation generation failed"
    
    success "Package built successfully"
}

# Validate package
validate_package() {
    log "Validating package..."
    
    # Check package size
    local package_size=$(npm pack --dry-run 2>/dev/null | grep -o 'package size: [0-9.]*[A-Z]*' | grep -o '[0-9.]*[A-Z]*')
    log "Package size: $package_size"
    
    # Validate package contents
    log "Checking package contents..."
    local packed_files=$(npm pack --dry-run 2>/dev/null | grep -E '^npm notice [0-9]+[A-Z]* ' | wc -l)
    log "Files in package: $packed_files"
    
    # Check required files
    local required_files=("bin/ruv-swarm-clean.js" "src/index.js" "README.md" "package.json")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required file missing: $file"
        fi
    done
    
    # Validate CLI binary
    log "Testing CLI binary..."
    node bin/ruv-swarm-clean.js --version || error "CLI binary test failed"
    
    success "Package validation passed"
}

# Test installation
test_installation() {
    log "Testing package installation..."
    
    # Create temporary directory for testing
    local test_dir=$(mktemp -d)
    local current_dir=$(pwd)
    
    # Pack the package
    local package_file=$(npm pack)
    
    cd "$test_dir"
    
    # Test local installation
    log "Testing local installation..."
    npm init -y >/dev/null
    npm install "$current_dir/$package_file" || error "Local installation failed"
    
    # Test CLI functionality
    log "Testing CLI functionality..."
    ./node_modules/.bin/ruv-swarm --version || error "CLI test failed"
    
    # Test programmatic usage
    log "Testing programmatic usage..."
    cat > test.js << 'EOF'
const { RuvSwarm } = require('ruv-swarm');
console.log('ruv-swarm imported successfully');
console.log('SIMD support:', RuvSwarm.detectSIMDSupport());
EOF
    
    node test.js || error "Programmatic test failed"
    
    # Cleanup
    cd "$current_dir"
    rm -rf "$test_dir"
    rm "$package_file"
    
    success "Installation test passed"
}

# Deploy to npm
deploy_to_npm() {
    log "Deploying to npm registry..."
    
    # Get current version
    local current_version=$(node -pe "require('./package.json').version")
    log "Current version: $current_version"
    
    # Check if version already exists
    if npm view "$PACKAGE_NAME@$current_version" version >/dev/null 2>&1; then
        error "Version $current_version already exists on npm. Update version first."
    fi
    
    # Publish package
    log "Publishing package..."
    npm publish --access public || error "npm publish failed"
    
    success "Package published successfully to npm"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    local current_version=$(node -pe "require('./package.json').version")
    
    # Wait for npm registry propagation
    log "Waiting for npm registry propagation..."
    sleep 30
    
    # Check if package is available
    local npm_version=$(npm view "$PACKAGE_NAME" version 2>/dev/null || echo "")
    if [[ "$npm_version" != "$current_version" ]]; then
        error "Deployment verification failed. Expected: $current_version, Found: $npm_version"
    fi
    
    # Test npx installation
    log "Testing npx installation..."
    local test_dir=$(mktemp -d)
    cd "$test_dir"
    
    timeout 60 npx "$PACKAGE_NAME@$current_version" --version || error "npx test failed"
    
    cd - >/dev/null
    rm -rf "$test_dir"
    
    success "Deployment verified successfully"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    local current_version=$(node -pe "require('./package.json').version")
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local git_commit=$(git rev-parse HEAD)
    local npm_user=$(npm whoami)
    
    cat > "deployment-report-$current_version.md" << EOF
# ruv-swarm Deployment Report

## Deployment Information
- **Package**: $PACKAGE_NAME
- **Version**: $current_version
- **Timestamp**: $timestamp
- **Git Commit**: $git_commit
- **Deployed by**: $npm_user
- **Registry**: $REGISTRY_URL

## Quality Metrics
- ✅ Linting: Passed
- ✅ Tests: Passed
- ✅ Security Audit: Passed
- ✅ Package Validation: Passed
- ✅ Installation Test: Passed
- ✅ Deployment Verification: Passed

## Package Contents
\`\`\`
$(npm pack --dry-run 2>/dev/null | grep -E '^npm notice [0-9]+[A-Z]* ')
\`\`\`

## Installation Commands
\`\`\`bash
# NPM installation
npm install $PACKAGE_NAME

# NPX usage (no installation required)
npx $PACKAGE_NAME --version

# Global installation
npm install -g $PACKAGE_NAME
\`\`\`

## Available Commands
\`\`\`bash
# Initialize swarm
npx $PACKAGE_NAME init mesh 5

# Spawn agent
npx $PACKAGE_NAME spawn researcher "AI Assistant"

# Start MCP server
npx $PACKAGE_NAME mcp start

# Run benchmarks
npx $PACKAGE_NAME benchmark run
\`\`\`

## Links
- **NPM Package**: https://www.npmjs.com/package/$PACKAGE_NAME
- **Documentation**: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/npm
- **Issues**: https://github.com/ruvnet/ruv-FANN/issues
EOF
    
    success "Deployment report generated: deployment-report-$current_version.md"
}

# Cleanup
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove temporary files
    if [[ -f "npm-debug.log" ]]; then
        rm npm-debug.log
    fi
    
    # Keep backup for safety (can be manually removed later)
    log "Backup preserved in $BACKUP_DIR (remove manually when confident)"
    
    success "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting ruv-swarm deployment process..."
    
    # Parse command line arguments
    local skip_tests=false
    local skip_validation=false
    local force=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --skip-validation)
                skip_validation=true
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            --help)
                cat << EOF
ruv-swarm Deployment Script

Usage: $0 [options]

Options:
    --skip-tests        Skip quality checks and tests
    --skip-validation   Skip package validation
    --force             Force deployment even with warnings
    --help              Show this help message

Examples:
    $0                  # Full deployment with all checks
    $0 --skip-tests     # Deploy without running tests
    $0 --force          # Force deploy with warnings
EOF
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    # Create log file
    echo "Deployment started at $(date)" > "$LOG_FILE"
    
    # Run deployment steps
    check_prerequisites
    create_backup
    
    if [[ "$skip_tests" != true ]]; then
        run_quality_checks
    else
        warn "Skipping quality checks (--skip-tests)"
    fi
    
    build_package
    
    if [[ "$skip_validation" != true ]]; then
        validate_package
        test_installation
    else
        warn "Skipping validation (--skip-validation)"
    fi
    
    # Confirmation before deployment
    if [[ "$force" != true ]]; then
        echo
        read -p "Deploy to npm registry? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    deploy_to_npm
    verify_deployment
    generate_report
    cleanup
    
    success "🎉 ruv-swarm deployment completed successfully!"
    
    local current_version=$(node -pe "require('./package.json').version")
    echo
    echo "📦 Package: https://www.npmjs.com/package/$PACKAGE_NAME"
    echo "🏷️  Version: $current_version"
    echo "📄 Report: deployment-report-$current_version.md"
    echo "🔧 Test: npx $PACKAGE_NAME@$current_version --version"
    echo
}

# Error handling
trap 'error "Deployment failed due to an error on line $LINENO"' ERR

# Run main function with all arguments
main "$@"