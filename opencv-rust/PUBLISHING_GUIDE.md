# Publishing OpenCV Rust to Crates.io

This guide explains how to publish the OpenCV Rust crates to crates.io using the .env configuration.

## Prerequisites

1. **Crates.io Account**: Create an account at https://crates.io
2. **API Token**: Generate an API token from https://crates.io/settings/tokens
3. **Cargo installed**: Ensure you have Rust and Cargo installed

## Step 1: Configure Environment

Edit the `.env` file in the root directory and update these values:

```bash
# Replace with your actual crates.io API token
CARGO_REGISTRY_TOKEN=your_actual_crates_io_token_here

# Set to false for actual publishing
PUBLISH_DRY_RUN=false
```

## Step 2: Verify Crate Readiness

Run a dry-run first to ensure everything is configured correctly:

```bash
cd /workspaces/ruv-FANN/opencv-rust
PUBLISH_DRY_RUN=true ./publish.sh
```

## Step 3: Publish to Crates.io

Once the dry-run succeeds, publish the crates:

```bash
# Load environment variables and publish
cd /workspaces/ruv-FANN/opencv-rust
source ../.env
./publish.sh
```

Alternatively, publish manually with proper token:

```bash
# Set token
export CARGO_REGISTRY_TOKEN="your_actual_token"

# Navigate to opencv-rust directory
cd /workspaces/ruv-FANN/opencv-rust

# Login to cargo
cargo login $CARGO_REGISTRY_TOKEN

# Publish in dependency order
cargo publish -p opencv-sys --allow-dirty
sleep 30  # Wait to avoid rate limiting

cargo publish -p opencv-core --allow-dirty
sleep 30

cargo publish -p opencv-imgproc --allow-dirty
sleep 30

cargo publish -p opencv-imgcodecs --allow-dirty
sleep 30

cargo publish -p opencv-ml --allow-dirty
sleep 30

cargo publish -p opencv-wasm --allow-dirty
sleep 30

cargo publish -p opencv-sdk --allow-dirty
```

## Step 4: Verify Publication

After publishing, verify your crates are available:

- Visit https://crates.io/crates/opencv-rust
- Check https://docs.rs/opencv-rust for documentation
- Test installation: `cargo add opencv-rust`

## Important Notes

1. **Rate Limiting**: Crates.io has rate limits. The script includes 30-second delays between publishes.
2. **Version Conflicts**: If a version already exists, increment the version in Cargo.toml files.
3. **Dependencies**: Ensure all workspace dependencies are published before dependent crates.
4. **Naming**: The crate names must be unique on crates.io. You may need to use prefixed names like `ruv-opencv-core`.

## Troubleshooting

### "crate name already taken"
The name `opencv-rust` or related names might already be taken. Consider using:
- `ruv-opencv`
- `ruv-opencv-rust`
- `opencv-fann`

Update all Cargo.toml files with the new names before publishing.

### "failed to verify package tarball"
Ensure all files are properly included in the package:
```bash
cargo package --list
```

### "missing required metadata"
Verify all Cargo.toml files have:
- name
- version
- authors
- license
- description
- repository

## Security Notes

- **Never commit** your actual CARGO_REGISTRY_TOKEN to git
- Use environment variables or CI/CD secrets for tokens
- Consider using a dedicated publishing token with limited scope

---

For questions or issues, please open an issue at https://github.com/ruvnet/ruv-FANN/issues