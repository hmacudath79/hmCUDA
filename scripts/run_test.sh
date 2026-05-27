#!/usr/bin/env bash
# Full integration test - builds, starts vhost, and prepares for VM testing
# Usage: ./run_test.sh [sample_name]

# Source common library
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
SAMPLE="${1:-smallTest}"

print_header "hmCUDA Full Integration Test"
log_info "Sample: $SAMPLE"
echo ""

# Step 1: Build
log_step "Building project..."
if ! build_project; then
    log_error "Build failed, cannot continue"
    exit 1
fi

# Check build outputs
if ! check_build_output; then
    log_error "Some build outputs are missing"
    exit 1
fi

# Step 2: Sync to share
log_step "Syncing files to share directory..."
"$SCRIPT_DIR/sync_to_share.sh"
log_success "Files synced"

# Step 3: Restart vhost daemon
log_step "Starting vhost daemon..."
vhost_restart

if ! vhost_is_running; then
    log_error "Vhost daemon failed to start"
    exit 1
fi

# Step 4: Print instructions
cat << 'INSTRUCTIONS'

=========================================
Host Setup Complete!
=========================================

INSTRUCTIONS

echo "✓ Project built"
echo "✓ Files synced to share/"
echo "✓ Vhost daemon running (PID: $(pgrep -f vhost_user_hmcuda))"
echo "✓ Socket ready: $VHOST_SOCKET"
echo ""

cat << 'INSTRUCTIONS'
=========================================
Next Steps - Start VM:
=========================================

1. In another terminal, start QEMU:
INSTRUCTIONS
echo "   cd $SCRIPT_DIR"
cat << 'INSTRUCTIONS'
   ./qemu_boot.sh

2. In the VM, run the test:
   sudo mkdir -p /mnt/host
   sudo mount -t virtiofs hostshare /mnt/host
   cd /mnt/host
INSTRUCTIONS
echo "   sudo HMCUDA_DEBUG=3 bash test_in_vm.sh $SAMPLE"
cat << 'INSTRUCTIONS'

   The VM kernel must already include/install virtio_hmcuda. To load a specific
   module artifact instead, set HMCUDA_DRIVER_KO=/path/to/virtio_hmcuda.ko.

Or use SSH (if configured):
   ssh -p 2222 fedora@localhost
   # Then run the commands above

=========================================
Monitoring:
=========================================

INSTRUCTIONS
echo "Host logs:  tail -f $VHOST_LOG"
cat << 'INSTRUCTIONS'
VM logs:    sudo dmesg | tail -50

=========================================
INSTRUCTIONS

# Keep vhost running
log_info "Vhost daemon is running. Press Ctrl+C to stop."
log_info "You can check status with: ./check_vhost.sh status"
echo ""

# Wait for vhost process
wait $(pgrep -f vhost_user_hmcuda)
