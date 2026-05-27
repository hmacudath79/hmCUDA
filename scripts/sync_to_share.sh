#!/usr/bin/env bash
# Sync build outputs and sources to share directory for VM access
# Usage: ./sync_to_share.sh

set -euo pipefail

# Source common library
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/common.sh"

print_header "Syncing to Share Directory"

# Create share directories
mkdir -p "$SHARE_DIR"/{runtime,sample}
rm -rf "$SHARE_DIR/hmcuda_driver"

# Copy runtime library (libcudart.so)
log_step "Syncing runtime library..."
if [ -f "$RUNTIME_LIB" ]; then
    rsync -a "$BUILD_DIR/guest/intercept/libcudart.so"* "$SHARE_DIR/runtime/"
    log_success "Runtime library synced ($(ls -lh $RUNTIME_LIB | awk '{print $5}'))"
else
    log_error "Runtime library not found: $RUNTIME_LIB"
fi

# Copy driver API library (libcuda.so)
log_step "Syncing driver API library..."
DRIVER_LIB="$BUILD_DIR/guest/intercept/libcuda.so.1.0.0"
if [ -f "$DRIVER_LIB" ]; then
    rsync -a "$BUILD_DIR/guest/intercept/libcuda.so"* "$SHARE_DIR/runtime/"
    log_success "Driver API library synced ($(ls -lh $DRIVER_LIB | awk '{print $5}'))"
else
    log_warn "Driver API library not found: $DRIVER_LIB (skipped)"
fi

# Copy NVML stub library (libnvidia-ml.so)
log_step "Syncing NVML stub library..."
NVML_LIB="$BUILD_DIR/guest/intercept/libnvidia-ml.so.1.0.0"
if [ -f "$NVML_LIB" ]; then
    rsync -a "$BUILD_DIR/guest/intercept/libnvidia-ml.so"* "$SHARE_DIR/runtime/"
    log_success "NVML stub library synced ($(ls -lh $NVML_LIB | awk '{print $5}'))"
else
    log_warn "NVML stub library not found: $NVML_LIB (skipped)"
fi

# Copy all common headers to runtime directory
rsync -a "$PROJECT_ROOT/common/include/"*.h "$SHARE_DIR/runtime/"

# Copy samples
log_step "Syncing sample binaries..."
sample_count=0
for sample in "$BUILD_DIR/sample"/*; do
    if [ -f "$sample" ] && [ -x "$sample" ]; then
        sample_name=$(basename "$sample")
        rsync -a "$sample" "$SHARE_DIR/sample/"
        log_success "  $sample_name ($(ls -lh $sample | awk '{print $5}'))"
        ((++sample_count))
    fi
done

if [ $sample_count -eq 0 ]; then
    log_error "No sample binaries found in $BUILD_DIR/sample"
else
    log_success "Synced $sample_count sample(s)"
fi

# Sync nvbandwidth binary
log_step "Syncing nvbandwidth..."
NVBW_BIN="$BUILD_DIR/third_party/nvbandwidth/nvbandwidth"
if [ -f "$NVBW_BIN" ]; then
    rsync -a "$NVBW_BIN" "$SHARE_DIR/sample/"
    log_success "nvbandwidth synced ($(ls -lh "$NVBW_BIN" | awk '{print $5}'))"
else
    log_warn "nvbandwidth binary not found at $NVBW_BIN (skipped)"
fi

# Copy cuBLAS libraries (if samples need them)
log_step "Checking for cuBLAS libraries..."
CUDA_CUBLAS_LIB=$(find "$CUDA_HOME" /usr/local/cuda* -name "libcublas.so*" 2>/dev/null | head -1 || true)
CUDA_LIB_DIR=""
if [ -n "$CUDA_CUBLAS_LIB" ]; then
    CUDA_LIB_DIR=$(dirname "$CUDA_CUBLAS_LIB")
fi
if [ -n "$CUDA_LIB_DIR" ] && [ -d "$CUDA_LIB_DIR" ]; then
    rsync -aL "$CUDA_LIB_DIR"/libcublas.so* "$SHARE_DIR/runtime/" 2>/dev/null || true
    rsync -aL "$CUDA_LIB_DIR"/libcublasLt.so* "$SHARE_DIR/runtime/" 2>/dev/null || true
    log_success "cuBLAS libraries synced"
else
    log_warn "cuBLAS libraries not found (optional)"
fi

# Copy VM test script and common library
log_step "Syncing test scripts..."
rsync -a "$SCRIPT_DIR/test_in_vm.sh" "$SHARE_DIR/"
rsync -a "$SCRIPT_DIR/common.sh" "$SHARE_DIR/"
log_success "Test scripts synced"

# Summary
echo ""
print_separator
echo "Sync Summary"
print_separator
echo "Driver:  built in-tree by the external Linux kernel project"
echo "Runtime: $SHARE_DIR/runtime/"
echo "Samples: $SHARE_DIR/sample/ ($sample_count binaries)"
echo "Scripts: $SHARE_DIR/test_in_vm.sh"
echo ""
echo "To access in VM:"
echo "  sudo mkdir -p /mnt/host"
echo "  sudo mount -t virtiofs hostshare /mnt/host"
echo "  ls -l /mnt/host/"
echo ""
