#!/usr/bin/env bash
# Keep the legacy entry point while using the same verified release path.
set -euo pipefail
release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$release_root/publish.sh" "$@"
