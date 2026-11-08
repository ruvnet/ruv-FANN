#!/usr/bin/env bash
# Validate first. Registry mutations require the explicit --publish flag.
set -euo pipefail

release_mode=dry-run
case "${1:---dry-run}" in
  --dry-run) ;;
  --publish) release_mode=publish ;;
  *) echo 'Usage: publish.sh [--dry-run|--publish]' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo 'Usage: publish.sh [--dry-run|--publish]' >&2
  exit 2
fi

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$release_root"

# Use Cargo's configured credential provider. Never source executable .env files,
# print token values, persist credentials with cargo login, or enable CUDA
# without the required native OpenCV build. Tests and package verification are hard gates.
cargo test --workspace --all-targets --locked
cargo check --workspace --exclude opencv-sys --all-features --locked
cargo check -p opencv-sys --locked
# Package the whole workspace together so Cargo verifies normalized local
# dependencies in its staging registry before any public registry write.
cargo package --workspace --locked
if [[ "$release_mode" == publish ]]; then
  for release_crate in opencv-core opencv-wasm opencv-sdk opencv-sys; do
    cargo publish -p "$release_crate" --locked
  done
fi
if [[ "$release_mode" == publish ]]; then
  echo 'Cargo reported successful publication for all four workspace crates.'
else
  echo 'Validation completed. No packages were published.'
fi
