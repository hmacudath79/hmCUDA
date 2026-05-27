#!/usr/bin/env bash
# Main development helper script for hmCUDA
# Provides a menu of common development tasks

# Source common library
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/common.sh"

KNOWN_TARGETS=(
    hmcuda_runtime
    hmcuda_driver
    hmcuda_nvml
    vhost_user_hmcuda
    nvbandwidth
    matrixMul
    smallTest
)

show_help() {
    cat << 'EOF'
hmCUDA Development Helper
=========================

Usage: ./dev.sh <command> [args]

Commands:
  build [target]      Build all targets, or a specific Make target
  clean [target]      Clean all Make outputs, or clean one known target group
  rebuild [target]    Clean then build (all, or specific target with --clean-first)

  vhost start [level]          Start vhost daemon (log level: 0=error 1=warn 2=info 3=debug, default 2)
  vhost stop                   Stop vhost daemon
  vhost restart [level]        Restart vhost daemon
  vhost reload [target]        Rebuild Make target + restart (QEMU auto-reconnects)
  vhost status                 Check vhost status
  vhost logs                   Show vhost logs (tail -f)

  sync                Sync files to share directory
  sync-driver [dir]   Copy driver sources into Linux tree (default: ~/linux/drivers/misc/hmcuda)
  initramfs           Pack busybox + guest CUDA programs into initramfs
  vm                  Start QEMU VM
  test [sample]       Full integration test (build + sync + vhost + instructions)

  status              Show complete system status

Known Make targets:
  hmcuda_runtime      libcudart.so.<CUDA major>
  hmcuda_driver       libcuda.so.1
  hmcuda_nvml         libnvidia-ml.so.1
  vhost_user_hmcuda   Host vhost-user daemon
  nvbandwidth         GPU bandwidth benchmark
  matrixMul           Matrix multiplication sample
  smallTest           Correctness test suite

Examples:
  ./dev.sh build                        # Build all targets
  ./dev.sh build vhost_user_hmcuda      # Build only the vhost daemon
  ./dev.sh clean                        # Remove entire build directory
  ./dev.sh clean hmcuda_runtime         # Remove runtime object/library outputs
  ./dev.sh rebuild                      # Full clean + rebuild
  ./dev.sh rebuild hmcuda_driver        # Rebuild only hmcuda_driver
  ./dev.sh vhost restart                # Restart vhost daemon
  ./dev.sh test smallTest               # Run full test with smallTest
  ./dev.sh sync                         # Sync to share directory
  ./dev.sh sync-driver                  # Copy driver into ~/linux/drivers/misc/hmcuda
  ./dev.sh initramfs                    # Pack build/hmcuda-initramfs.cpio.gz
  ./dev.sh status                       # Show system status

EOF
}

cmd_build() {
    local target="${1:-}"
    if [ -n "$target" ]; then
        print_header "Building Target: $target"
        make -C "$PROJECT_ROOT" "$target" -j"$(nproc)"
        log_success "Target '$target' built successfully"
    else
        print_header "Building Project"
        build_project
        check_build_output
    fi
}

_target_clean() {
    local target="$1"
    case "$target" in
        hmcuda_runtime)
            rm -f "$BUILD_DIR/guest/intercept/hmcuda_runtime.o" "$BUILD_DIR/guest/intercept/libcudart.so"*
            ;;
        hmcuda_driver)
            rm -f "$BUILD_DIR/guest/intercept/hmcuda_driver.o" "$BUILD_DIR/guest/intercept/libcuda.so"*
            ;;
        hmcuda_nvml|hmcuda_nvml_stub)
            rm -f "$BUILD_DIR/guest/intercept/hmcuda_nvml_stub.o" "$BUILD_DIR/guest/intercept/libnvidia-ml.so"*
            ;;
        vhost_user_hmcuda)
            make -C "$PROJECT_ROOT" clean-host
            ;;
        matrixMul)
            rm -f "$BUILD_DIR/sample/matrixMul"
            ;;
        smallTest)
            rm -f "$BUILD_DIR/sample/smallTest"
            ;;
        guest|host|samples)
            make -C "$PROJECT_ROOT" "clean-$target"
            ;;
        driver)
            log_warn "virtio_hmcuda is built in-tree by the external Linux kernel project; nothing to clean here"
            ;;
        *)
            log_warn "No specific clean rule for '$target'; running make clean-$target if available"
            make -C "$PROJECT_ROOT" "clean-$target"
            ;;
    esac
    log_success "Target '$target' cleaned"
}

cmd_clean() {
    local target="${1:-}"
    if [ -n "$target" ]; then
        print_header "Cleaning Target: $target"
        if [ ! -d "$BUILD_DIR" ]; then
            log_warn "Build directory does not exist, nothing to clean"
            return 0
        fi
        _target_clean "$target"
    else
        print_header "Cleaning Build Directory"
        make -C "$PROJECT_ROOT" clean
        log_success "Build outputs removed"
    fi
}

cmd_rebuild() {
    local target="${1:-}"
    if [ -n "$target" ]; then
        print_header "Rebuilding Target: $target"
        _target_clean "$target"
        make -C "$PROJECT_ROOT" "$target" -j"$(nproc)"
        log_success "Target '$target' rebuilt successfully"
    else
        cmd_clean
        cmd_build
    fi
}

cmd_sync() {
    print_header "Syncing to Share"
    "$SCRIPT_DIR/sync_to_share.sh"
}

cmd_sync_driver() {
    print_header "Syncing Driver to Linux Kernel Tree"
    "$SCRIPT_DIR/sync_driver_to_kernel.sh" "$@"
}

cmd_initramfs() {
    print_header "Packing Initramfs"
    "$SCRIPT_DIR/pack_initramfs.sh" "$@"
}

cmd_vm() {
    print_header "Starting QEMU VM"
    cd "$SCRIPT_DIR"
    exec ./qemu_boot.sh
}

cmd_test() {
    local sample="${1:-smallTest}"
    exec "$SCRIPT_DIR/run_test.sh" "$sample"
}

cmd_status() {
    print_header "System Status"

    # Build status
    echo "Build Status:"
    if [ -d "$BUILD_DIR" ]; then
        log_success "Build directory exists"
        check_build_output 2>/dev/null && log_success "All outputs present" || log_warn "Some outputs missing"
    else
        log_fail "Build directory not found"
    fi
    echo ""

    # Vhost status
    vhost_status

    # Share status
    echo "Share Directory:"
    if [ -d "$SHARE_DIR" ]; then
        log_success "Share directory exists: $SHARE_DIR"
        local sample_files=$(ls -1 "$SHARE_DIR/sample" 2>/dev/null | wc -l)
        echo "  Driver: built in-tree by external Linux kernel project"
        echo "  Sample binaries: $sample_files"
    else
        log_warn "Share directory not found"
    fi
    echo ""
}

cmd_vhost() {
    local subcmd="${1:-status}"
    shift || true

    case "$subcmd" in
        logs)
            if [ -f "$VHOST_LOG" ]; then
                tail -f "$VHOST_LOG"
            else
                log_error "Log file not found: $VHOST_LOG"
                exit 1
            fi
            ;;
        *)
            exec "$SCRIPT_DIR/check_vhost.sh" "$subcmd" "$@"
            ;;
    esac
}

# Main command dispatcher
CMD="${1:-help}"
shift || true

case "$CMD" in
    build)
        cmd_build "$@"
        ;;
    clean)
        cmd_clean "$@"
        ;;
    rebuild)
        cmd_rebuild "$@"
        ;;
    sync)
        cmd_sync
        ;;
    sync-driver)
        cmd_sync_driver "$@"
        ;;
    initramfs)
        cmd_initramfs "$@"
        ;;
    vm)
        cmd_vm
        ;;
    test)
        cmd_test "$@"
        ;;
    status)
        cmd_status
        ;;
    vhost)
        cmd_vhost "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $CMD"
        echo ""
        show_help
        exit 1
        ;;
esac
