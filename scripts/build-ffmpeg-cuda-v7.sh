#!/bin/bash
# Build ffmpeg-cuda-v7 (or any N version) with CUDA + NVENC + libass.
# Single binary that handles BOTH concat + burn with h264_nvenc.
#
# Usage: ./build-ffmpeg-cuda-v7.sh [version]
#   version: ffmpeg release tag (default: n9.0.1)
#
# Requires: apt install libass-dev libfreetype-dev libfontconfig-dev
#                  libfribidi-dev libharfbuzz-dev libx264-dev libx265-dev
#           nvidia-cuda-toolkit (provides nvcc)
#
# Time: ~15-30 min on 32 cores, ~1.1GB disk during build

set -e

VERSION="${1:-n9.0.1}"
INSTALL_PREFIX="/opt/ffmpeg-cuda-v7"
BUILD_DIR="/opt/build-ffmpeg-cuda-v7"
JOBS=$(nproc)

echo "=== Building ffmpeg ${VERSION} -> ${INSTALL_PREFIX} ==="
echo "    CPUs: $JOBS"
echo ""

# Cleanup previous build
if [ -d "$BUILD_DIR" ]; then
    echo "Removing old build dir $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone source
echo "[1/4] Cloning ffmpeg ${VERSION}..."
git clone --depth 1 --branch "$VERSION" https://github.com/FFmpeg/FFmpeg.git ffmpeg-src
cd ffmpeg-src

# Configure
echo "[2/4] Configuring..."
./configure \
    --prefix="$INSTALL_PREFIX" \
    --enable-cuda --enable-cuda-nvcc --nvcc=/usr/bin/nvcc \
    --enable-nvenc --enable-nvdec \
    --enable-nonfree \
    --enable-libass --enable-libfreetype --enable-fontconfig \
    --enable-libfribidi --enable-libharfbuzz \
    --enable-libx264 --enable-libx265 \
    --enable-gpl \
    --enable-cuda-llvm \
    --enable-runtime-cpudetect \
    --extra-cflags=-O3 \
    2>&1 | tail -5

# Compile
echo "[3/4] Compiling (this takes 15-30 min)..."
make -j"$JOBS" 2>&1 | tail -5

# Install
echo "[4/4] Installing..."
make install 2>&1 | tail -3

# Verify
echo ""
echo "=== Verification ==="
"$INSTALL_PREFIX/bin/ffmpeg" -version | head -1
echo ""
echo "Subtitles filter (libass):"
"$INSTALL_PREFIX/bin/ffmpeg" -hide_banner -h filter=subtitles 2>&1 | head -3
echo ""
echo "NVENC encoders:"
"$INSTALL_PREFIX/bin/ffmpeg" -hide_banner -encoders 2>&1 | grep -E "h264_nvenc|hevc_nvenc" | head -3
echo ""
echo "Linked libs:"
ldd "$INSTALL_PREFIX/bin/ffmpeg" | grep -iE "ass|freetype|fontconfig|fribidi|harfbuzz" | head -5
echo ""
echo "=== Done. Binary at: $INSTALL_PREFIX/bin/ffmpeg ==="
echo "Clean up build dir with: sudo rm -rf $BUILD_DIR"
echo "Disk freed: $(du -sh $BUILD_DIR 2>/dev/null | cut -f1)"
