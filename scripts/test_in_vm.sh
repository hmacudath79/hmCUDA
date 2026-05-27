#!/usr/bin/env bash
# Test script to run INSIDE the Fedora VM
# Usage: sudo bash test_in_vm.sh [sample_name]

# Source common library (if available, otherwise define minimal functions)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/common.sh" ]; then
    source "$SCRIPT_DIR/common.sh"
else
    # Minimal fallback functions
    COLOR_GREEN='\033[0;32m'
    COLOR_RED='\033[0;31m'
    COLOR_RESET='\033[0m'

    log_step() { echo -e "${COLOR_GREEN}[STEP]${COLOR_RESET} $*"; }
    log_error() { echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*" >&2; }
    log_success() { echo -e "${COLOR_GREEN}[✓]${COLOR_RESET} $*"; }
    print_header() { echo ""; echo "========================================="; echo "$*"; echo "========================================="; echo ""; }
fi

# Parse arguments
SAMPLE="${1:-smallTest}"
shift || true
SAMPLE_ARGS=("$@")   # remaining args are forwarded to the sample binary
AVAILABLE_SAMPLES="matrixMul smallTest nvbandwidth"

# Help
if [ "$SAMPLE" = "-h" ] || [ "$SAMPLE" = "--help" ]; then
    echo "Usage: sudo bash $0 [sample_name] [sample_args...]"
    echo ""
    echo "Available samples: $AVAILABLE_SAMPLES"
    echo "Default: smallTest"
    echo ""
    echo "Log levels (HMCUDA_DEBUG):"
    echo "  0 = Silent"
    echo "  1 = Errors only"
    echo "  2 = Warnings"
    echo "  3 = Info (default)"
    echo "  4 = Debug (verbose)"
    echo "  5 = Timing only (per-ioctl latency)"
    echo ""
    echo "Examples:"
    echo "  sudo bash $0 smallTest"
    echo "  sudo HMCUDA_DEBUG=4 bash $0 smallTest"
    echo "  sudo HMCUDA_DEBUG=5 bash $0 smallTest"
    echo "  sudo bash $0 nvbandwidth --help"
    echo "  sudo bash $0 nvbandwidth --list"
    echo "  sudo bash $0 nvbandwidth --testcase host_to_device_memcpy_ce"
    echo "  sudo bash $0 nvbandwidth -t 0 -t 1"
    exit 0
fi

print_header "hmCUDA VM Test - $SAMPLE${SAMPLE_ARGS[*]:+ ${SAMPLE_ARGS[*]}}"

# Set default log level if not set
export HMCUDA_DEBUG="${HMCUDA_DEBUG:-4}"
log_step "Log level: HMCUDA_DEBUG=$HMCUDA_DEBUG"

# Step 1: Check if running in VM
if [ ! -d "/mnt/host/runtime" ]; then
    log_error "Cannot find /mnt/host/runtime"
    echo ""
    echo "Please mount virtiofs first:"
    echo "  sudo mkdir -p /mnt/host"
    echo "  sudo mount -t virtiofs hostshare /mnt/host"
    exit 1
fi

log_success "VirtIO filesystem mounted"

# Step 2: Load driver
log_step "Loading driver..."
DRIVER_LOG_LEVEL="${HMCUDA_DRIVER_LOG:-0}"
if [ -c "/dev/virtio-hmcuda" ]; then
    log_success "Device already ready"
elif [ -n "${HMCUDA_DRIVER_KO:-}" ]; then
    sudo rmmod virtio_hmcuda 2>/dev/null || true
    if sudo insmod "$HMCUDA_DRIVER_KO" hmcuda_log_level="$DRIVER_LOG_LEVEL"; then
        log_success "Driver loaded from $HMCUDA_DRIVER_KO"
    else
        log_error "Failed to load driver from HMCUDA_DRIVER_KO=$HMCUDA_DRIVER_KO"
        sudo dmesg | tail -20
        exit 1
    fi
elif sudo modprobe virtio_hmcuda hmcuda_log_level="$DRIVER_LOG_LEVEL"; then
    log_success "Driver loaded via modprobe"
else
    log_error "Failed to load virtio_hmcuda"
    echo ""
    echo "The driver is expected to be built in-tree by the external Linux kernel project."
    echo "Install it into /lib/modules for modprobe, or set HMCUDA_DRIVER_KO=/path/to/virtio_hmcuda.ko."
    sudo dmesg | tail -20
    exit 1
fi

# Wait for device
sleep 1

if [ ! -c "/dev/virtio-hmcuda" ]; then
    log_error "Device /dev/virtio-hmcuda not created"
    sudo dmesg | tail -20
    exit 1
fi

log_success "Device ready: $(ls -l /dev/virtio-hmcuda | awk '{print $1, $5, $6, $10}')"

# Step 3: Check runtime library
if ! ls /mnt/host/runtime/libcudart.so* >/dev/null 2>&1; then
    log_error "Runtime library not found under /mnt/host/runtime/libcudart.so*"
    exit 1
fi

log_success "Runtime library found"

# Step 4: Check sample binary
if [ ! -f "/mnt/host/sample/$SAMPLE" ]; then
    log_error "Sample binary '$SAMPLE' not found at /mnt/host/sample/$SAMPLE"
    echo ""
    echo "Available samples:"
    ls -1 /mnt/host/sample/ 2>/dev/null | grep -v '\.' || echo "  (none found)"
    exit 1
fi

log_success "Sample binary found: $SAMPLE"

# Step 5: Run test
echo ""
print_header "Test Output"
cd /mnt/host/sample
chmod +x "$SAMPLE"

if sudo LD_LIBRARY_PATH=/mnt/host/runtime HMCUDA_DEBUG=$HMCUDA_DEBUG ./"$SAMPLE" "${SAMPLE_ARGS[@]}"; then
    TEST_RESULT="PASSED"
else
    TEST_RESULT="FAILED"
fi

# Step 6: Show results
echo ""
print_header "Test Result: $TEST_RESULT"

if [ "$TEST_RESULT" = "PASSED" ]; then
    log_success "Test completed successfully!"
else
    log_error "Test failed! Check logs below"
fi

echo ""
log_step "Recent kernel messages (dmesg):"
sudo dmesg | tail -20

echo ""
log_step "Host vhost logs:"
echo "  On host: tail -50 /tmp/vhost-hmcuda.log"

echo ""
print_separator
echo "Cleanup (optional):"
print_separator
echo "To unload driver: sudo rmmod virtio_hmcuda"
echo ""

# Exit with appropriate code
if [ "$TEST_RESULT" = "PASSED" ]; then
    exit 0
else
    exit 1
fi
