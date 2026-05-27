#!/usr/bin/env bash
# Manage vhost-user-hmcuda daemon
# Usage: ./check_vhost.sh [start|stop|restart|status]

# Source common library
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/common.sh"

# Parse command
CMD="${1:-restart}"
LOG_LEVEL="${2:-2}"   # 0=error 1=warn 2=info 3=debug
TARGET="${2:-vhost_user_hmcuda}"   # for reload: make target to rebuild

case "$CMD" in
    start)
        print_header "Starting Vhost Daemon"
        vhost_start "$LOG_LEVEL"
        ;;
    stop)
        print_header "Stopping Vhost Daemon"
        vhost_stop
        vhost_cleanup_socket
        ;;
    restart)
        print_header "Restarting Vhost Daemon"
        vhost_restart "$LOG_LEVEL"
        ;;
    reload)
        # Rebuild target then restart — QEMU reconnects automatically (reconnect=1)
        print_header "Reloading Vhost (rebuild + restart)"
        log_step "Building target: $TARGET"
        make -C "$PROJECT_ROOT" "$TARGET" -j"$(nproc)"
        log_success "Build done"
        vhost_restart
        log_info "QEMU will reconnect automatically (reconnect=1)"
        log_info "In the VM, reload the driver:"
        log_info "  sudo rmmod virtio_hmcuda && sudo modprobe virtio_hmcuda"
        ;;
    status)
        vhost_status
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|reload|status]"
        echo ""
        echo "Commands:"
        echo "  start   - Start the vhost daemon"
        echo "  stop    - Stop the vhost daemon"
        echo "  restart - Restart the vhost daemon (default)"
        echo "  reload [target] - Rebuild Make target then restart (default: vhost_user_hmcuda)"
        echo "  status  - Check daemon status"
        exit 1
        ;;
esac

# Show status if started/restarted/reloaded successfully
if [ "$CMD" = "start" ] || [ "$CMD" = "restart" ] || [ "$CMD" = "reload" ]; then
    vhost_status

    echo "You can now start QEMU with:"
    echo "  cd $SCRIPT_DIR && ./qemu_boot.sh"
    echo ""
fi
