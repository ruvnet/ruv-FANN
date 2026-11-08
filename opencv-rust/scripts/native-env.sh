#!/usr/bin/env bash
# Source this file with a relocated OpenCV prefix, or use system pkg-config.
# Example: source opencv-rust/scripts/native-env.sh /path/to/prefix
if [[ -n "${1:-${OPENCV_PREFIX:-}}" ]]; then
  export OPENCV_PREFIX="${1:-$OPENCV_PREFIX}"
  if [[ ! -d "$OPENCV_PREFIX/usr/include/opencv4" ]]; then
    echo 'OpenCV prefix must contain usr/include/opencv4' >&2
    return 1 2>/dev/null || exit 1
  fi
  export PKG_CONFIG_SYSROOT_DIR="$OPENCV_PREFIX"
  export PKG_CONFIG_PATH="$OPENCV_PREFIX/usr/lib/$(gcc -dumpmachine)/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  for resource in "$OPENCV_PREFIX"/usr/lib/llvm-*/lib/clang/*/include; do
    if [[ -f "$resource/stddef.h" ]]; then
      export BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS:-} -isystem \"$resource\""
      break
    fi
  done
  export LD_LIBRARY_PATH="$OPENCV_PREFIX/usr/lib/$(gcc -dumpmachine)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
pkg-config --atleast-version=4.4 opencv4
