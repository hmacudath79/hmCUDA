#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/common.sh"

QEMU=qemu-system-x86_64
KERNEL_IMAGE="${KERNEL_IMAGE:-$HOME/linux/arch/x86/boot/bzImage}"
INITRAMFS_IMAGE="${INITRAMFS_IMAGE:-$BUILD_DIR/hmcuda-initramfs.cpio.gz}"
QEMU_MEMORY="${QEMU_MEMORY:-4G}"
QEMU_CPUS="${QEMU_CPUS:-4}"
KERNEL_CMDLINE="${KERNEL_CMDLINE:-console=ttyS0 rdinit=/init panic=-1 virtio_hmcuda.hmcuda_log_level=1}"
QEMU_EXTRA_ARGS="${QEMU_EXTRA_ARGS:-}"
VHOST_USER_DEVICE="${VHOST_USER_DEVICE:-}"
DRY_RUN="${DRY_RUN:-0}"

qemu_has_device() {
    local dev="$1"
    $QEMU -device "$dev",help 2>&1 | grep -q "^$dev options:"
}

# ----------------------------------------
# KVM availability check
# ----------------------------------------
KVM_ARGS="-enable-kvm -cpu host"

if [ ! -e /dev/kvm ]; then
    log_warn "/dev/kvm not found — running without KVM (slow)"
    KVM_ARGS="-cpu max"
elif ! test -r /dev/kvm || ! test -w /dev/kvm; then
    log_warn "/dev/kvm not accessible by current user (group: $(stat -c %G /dev/kvm))"
    log_warn "Fix once with: sudo usermod -aG kvm \$USER  (then re-login)"
    KVM_ARGS="-cpu max"
else
    log_success "KVM available"
fi

# ----------------------------------------
# Check required files
# ----------------------------------------
if [ ! -f "$KERNEL_IMAGE" ]; then
    log_error "Kernel image not found: $KERNEL_IMAGE"
    log_error "Set KERNEL_IMAGE=/path/to/bzImage or build ~/linux/arch/x86/boot/bzImage"
    exit 1
fi

if [ ! -f "$INITRAMFS_IMAGE" ]; then
    log_error "Initramfs image not found: $INITRAMFS_IMAGE"
    log_error "Run: ./dev.sh initramfs"
    exit 1
fi

if [ ! -S "$VHOST_SOCKET" ]; then
    log_warn "Vhost socket not found: $VHOST_SOCKET"
    log_warn "Run: ./check_vhost.sh restart"
fi

if [ -z "$VHOST_USER_DEVICE" ]; then
    if qemu_has_device vhost-user-device-pci; then
        VHOST_USER_DEVICE="vhost-user-device-pci"
    elif qemu_has_device vhost-user-test-device-pci; then
        VHOST_USER_DEVICE="vhost-user-test-device-pci"
    else
        log_error "No generic vhost-user PCI device frontend found in $($QEMU --version | head -1)"
        log_error "Expected vhost-user-device-pci or vhost-user-test-device-pci."
        exit 1
    fi
fi

# ----------------------------------------
# Launch QEMU
# ----------------------------------------
print_header "Starting Direct-Boot QEMU VM"
log_info "KVM args: $KVM_ARGS"
log_info "Kernel:   $KERNEL_IMAGE"
log_info "Initramfs:$INITRAMFS_IMAGE"
log_info "Cmdline:  $KERNEL_CMDLINE"
log_info "Memory:   $QEMU_MEMORY"
log_info "CPUs:     $QEMU_CPUS"
log_info "Socket:   $VHOST_SOCKET"
log_info "Device:   $VHOST_USER_DEVICE"
echo ""

# QEMU_EXTRA_ARGS is intentionally word-split so callers can pass normal QEMU
# argument fragments, for example: QEMU_EXTRA_ARGS="-serial mon:stdio".
# shellcheck disable=SC2086
QEMU_CMD=(
  "$QEMU"
  $KVM_ARGS
  -m "$QEMU_MEMORY"
  -object "memory-backend-memfd,id=mem,size=$QEMU_MEMORY,share=on"
  -machine memory-backend=mem
  -smp "$QEMU_CPUS"
  -kernel "$KERNEL_IMAGE"
  -initrd "$INITRAMFS_IMAGE"
  -append "$KERNEL_CMDLINE"
  -nographic
  -chardev "socket,id=hmcuda0,path=$VHOST_SOCKET"
  -device "$VHOST_USER_DEVICE,chardev=hmcuda0,virtio-id=30"
  $QEMU_EXTRA_ARGS
)

if [ "$DRY_RUN" = "1" ]; then
    printf 'QEMU command:'
    printf ' %q' "${QEMU_CMD[@]}"
    printf '\n'
    exit 0
fi

"${QEMU_CMD[@]}"
