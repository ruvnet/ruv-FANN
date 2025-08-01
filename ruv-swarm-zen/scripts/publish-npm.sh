#!/bin/bash
# Script to publish NPM package
# Must be run after Rust crates are published

set -e

echo "📦 Publishing ruv-swarm NPM package..."

# Navigate to NPM directory
cd "$(dirname "$0")/../npm"

# Check if we're logged in to npm
if ! npm whoami >/dev/null 2>&1; then
    echo "❌ Please login to npm first with: npm login"
    exit 1
fi

# Build WASM artifacts
echo "🔨 Building WASM artifacts..."
npm run build:all

# Run tests
echo "🧪 Running tests..."
npm run test:all

# Run linting
echo "🔍 Running linter..."
npm run lint:check

# Verify package contents
echo "📋 Package contents:"
npm pack --dry-run

# Get current version
VERSION=$(node -p "require('./package.json').version")
echo "📌 Current version: $VERSION"

# Confirm publishing
read -p "Publish ruv-swarm@$VERSION to npm? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Publish to npm
npm publish --access public

echo "🎉 NPM package published successfully!"
echo "📝 View at: https://www.npmjs.com/package/ruv-swarm"