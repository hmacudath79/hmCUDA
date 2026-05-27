#!/usr/bin/env bash
# Build and install GCC 13 for CUDA nvcc host-compiler use.
#
# Defaults:
#   GCC_VERSION=13.4.0
#   PREFIX=/opt/gcc-13.4.0
#   LINK_PREFIX=/opt/gcc-13
#
# Usage:
#   scripts/install_gcc13.sh
#   PREFIX="$HOME/opt/gcc-13.4.0" LINK_PREFIX="$HOME/opt/gcc-13" scripts/install_gcc13.sh

set -euo pipefail

GCC_VERSION="${GCC_VERSION:-13.4.0}"
PREFIX="${PREFIX:-/opt/gcc-$GCC_VERSION}"
LINK_PREFIX="${LINK_PREFIX:-/opt/gcc-13}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/hmcuda-gcc13-build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 1)}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
DOWNLOAD_PREREQS="${DOWNLOAD_PREREQS:-1}"
SRC_URL="${SRC_URL:-https://ftp.gnu.org/gnu/gcc/gcc-$GCC_VERSION/gcc-$GCC_VERSION.tar.xz}"
BUILD_CFLAGS="${BUILD_CFLAGS:--g -O2}"
BUILD_CXXFLAGS="${BUILD_CXXFLAGS:--g -O2}"

need_sudo() {
    [ "${1#/opt/}" != "$1" ] || [ "${1#/usr/}" != "$1" ]
}

run_install() {
    if need_sudo "$PREFIX" || need_sudo "$LINK_PREFIX"; then
        sudo "$@"
    else
        "$@"
    fi
}

if [ "$INSTALL_DEPS" = "1" ]; then
    sudo dnf install -y \
        gcc gcc-c++ make binutils glibc-devel libstdc++-devel \
        gmp-devel mpfr-devel libmpc-devel isl-devel zlib-devel \
        flex bison diffutils tar xz curl
fi

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

archive="gcc-$GCC_VERSION.tar.xz"
src_dir="$BUILD_ROOT/gcc-$GCC_VERSION"
build_dir="$BUILD_ROOT/build-gcc-$GCC_VERSION"

if [ ! -f "$archive" ]; then
    curl -L "$SRC_URL" -o "$archive"
fi

if [ ! -d "$src_dir" ]; then
    tar -xf "$archive"
fi

if [ "$DOWNLOAD_PREREQS" = "1" ] && [ ! -f "$src_dir/.hmcuda-prereqs-ready" ]; then
    (cd "$src_dir" && ./contrib/download_prerequisites)
    touch "$src_dir/.hmcuda-prereqs-ready"
fi

if [ -f "$src_dir/libcody/configure" ] && [ ! -f "$src_dir/.hmcuda-libcody-cxx17-ready" ]; then
    sed -i \
        -e 's/#if __cplusplus != 201103/#if __cplusplus < 201103/g' \
        -e 's/#if __cplusplus > 201103/#if 0/g' \
        "$src_dir/libcody/configure"
    touch "$src_dir/.hmcuda-libcody-cxx17-ready"
fi

if [ -d "$src_dir/libcody" ] && [ ! -f "$src_dir/.hmcuda-libcody-char8-v2-ready" ]; then
    find "$src_dir/libcody" -type f \( -name '*.cc' -o -name '*.hh' -o -name '*.h' \) \
        -exec sed -i 's/u8"/"/g' {} +
    touch "$src_dir/.hmcuda-libcody-char8-v2-ready"
fi

if [ -f "$src_dir/gcc/system.h" ] && [ ! -f "$src_dir/.hmcuda-system-locale-ready" ]; then
    sed -i '/# include <type_traits>/a # include <locale>' "$src_dir/gcc/system.h"
    touch "$src_dir/.hmcuda-system-locale-ready"
fi

rm -rf "$build_dir"
mkdir -p "$build_dir"
cd "$build_dir"
log_file="$BUILD_ROOT/gcc-$GCC_VERSION-build.log"

CFLAGS="$BUILD_CFLAGS" \
CXXFLAGS="$BUILD_CXXFLAGS" \
CFLAGS_FOR_BUILD="$BUILD_CFLAGS" \
CXXFLAGS_FOR_BUILD="$BUILD_CXXFLAGS" \
"$src_dir/configure" \
    --prefix="$PREFIX" \
    --program-suffix=-13 \
    --enable-languages=c,c++ \
    --disable-multilib \
    --disable-bootstrap \
    --disable-nls \
    --disable-libsanitizer \
    --disable-libquadmath \
    --disable-libvtv \
    --disable-libstdcxx-pch \
    --with-system-zlib

if ! make -j"$JOBS" 2>&1 | tee "$log_file"; then
    echo "" >&2
    echo "GCC build failed. Full log: $log_file" >&2
    echo "Last 80 log lines:" >&2
    tail -80 "$log_file" >&2
    exit 1
fi

if ! run_install make install 2>&1 | tee -a "$log_file"; then
    echo "" >&2
    echo "GCC install failed. Full log: $log_file" >&2
    echo "Last 80 log lines:" >&2
    tail -80 "$log_file" >&2
    exit 1
fi

run_install ln -sfn "$PREFIX" "$LINK_PREFIX"

cat <<EOF

GCC $GCC_VERSION installed.

Use it for hmCUDA builds with:
  source scripts/cuda_env.sh 13.2 "$LINK_PREFIX"
  make clean
  make -j\$(nproc)

Compiler paths:
  $LINK_PREFIX/bin/gcc-13
  $LINK_PREFIX/bin/g++-13
EOF
