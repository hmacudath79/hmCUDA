#!/usr/bin/env bash
# Common library for hmCUDA scripts
# This file should be sourced, not executed directly

# Exit on error
set -euo pipefail

# ============================================
# Directory and Path Setup
# ============================================

# Get script directory (works when sourced)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
else
    SCRIPT_DIR="$(pwd)"
fi

PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
SHARE_DIR="$PROJECT_ROOT/share"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_SO_MAJOR="${CUDA_SO_MAJOR:-$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/^.*release \([0-9][0-9]*\)\..*$/\1/p' | head -1)}"
CUDA_SO_MAJOR="${CUDA_SO_MAJOR:-13}"
CUDA_VERSION="${CUDA_VERSION:-$CUDA_SO_MAJOR.0.0}"

# ============================================
# File Paths
# ============================================

VHOST_BIN="$BUILD_DIR/host/vhost/vhost_user_hmcuda"
VHOST_SOCKET="/tmp/vhost-hmcuda.sock"
VHOST_LOG="/tmp/vhost-hmcuda.log"

RUNTIME_LIB="$BUILD_DIR/guest/intercept/libcudart.so.$CUDA_SO_MAJOR"
SAMPLE_DIR="$BUILD_DIR/sample"

# ============================================
# Colors and Formatting
# ============================================

if [ -t 1 ]; then
    # Terminal supports colors
    COLOR_GREEN='\033[0;32m'
    COLOR_YELLOW='\033[1;33m'
    COLOR_RED='\033[0;31m'
    COLOR_BLUE='\033[0;34m'
    COLOR_RESET='\033[0m'
else
    # No color support
    COLOR_GREEN=''
    COLOR_YELLOW=''
    COLOR_RED=''
    COLOR_BLUE=''
    COLOR_RESET=''
fi

# ============================================
# Logging Functions
# ============================================

log_step() {
    echo -e "${COLOR_GREEN}[STEP]${COLOR_RESET} $*"
}

log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $*"
}

log_warn() {
    echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $*"
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*" >&2
}

log_success() {
    echo -e "${COLOR_GREEN}[✓]${COLOR_RESET} $*"
}

log_fail() {
    echo -e "${COLOR_RED}[✗]${COLOR_RESET} $*" >&2
}

print_separator() {
    echo "========================================="
}

print_header() {
    echo ""
    print_separator
    echo "$*"
    print_separator
    echo ""
}

# ============================================
# Vhost Daemon Management
# ============================================

vhost_is_running() {
    pgrep -f "vhost_user_hmcuda" > /dev/null
}

vhost_stop() {
    if vhost_is_running; then
        log_step "Stopping vhost daemon..."
        pkill -f "vhost_user_hmcuda"
        sleep 1
        log_success "Daemon stopped"
    else
        log_info "Vhost daemon is not running"
    fi
}

vhost_cleanup_socket() {
    if [ -S "$VHOST_SOCKET" ] || [ -e "$VHOST_SOCKET" ]; then
        log_step "Cleaning up stale socket..."
        rm -f "$VHOST_SOCKET"
        log_success "Socket removed"
    fi
}

vhost_start() {
    local log_level="${1:-2}"   # default INFO
    log_step "Starting vhost daemon (log level: $log_level)..."

    # Check binary exists
    if [ ! -f "$VHOST_BIN" ]; then
        log_error "Vhost binary not found at $VHOST_BIN"
        log_error "Run: make -C $PROJECT_ROOT vhost_user_hmcuda"
        return 1
    fi

    # Start daemon in background
    HMCUDA_LOG_LEVEL="$log_level" "$VHOST_BIN" "$VHOST_SOCKET" > "$VHOST_LOG" 2>&1 &
    local daemon_pid=$!

    # Wait for socket to be created
    log_info "Waiting for socket to be created..."
    local timeout=10
    for i in $(seq 1 $timeout); do
        if [ -S "$VHOST_SOCKET" ]; then
            log_success "Socket created successfully"
            log_success "Daemon started (PID: $daemon_pid)"
            log_info "Log file: $VHOST_LOG"
            return 0
        fi
        sleep 0.5
    done

    log_error "Socket was not created after ${timeout} seconds"
    log_error "Last 20 lines of log:"
    tail -20 "$VHOST_LOG" >&2
    return 1
}

vhost_restart() {
    local log_level="${1:-2}"
    vhost_stop
    vhost_cleanup_socket
    vhost_start "$log_level"
}

vhost_status() {
    echo ""
    print_separator
    echo "Vhost Daemon Status"
    print_separator

    if vhost_is_running; then
        local pid=$(pgrep -f "vhost_user_hmcuda")
        log_success "Running (PID: $pid)"
    else
        log_fail "Not running"
    fi

    if [ -S "$VHOST_SOCKET" ]; then
        log_success "Socket exists: $VHOST_SOCKET"
    else
        log_fail "Socket missing: $VHOST_SOCKET"
    fi

    if [ -f "$VHOST_LOG" ]; then
        log_info "Log file: $VHOST_LOG ($(wc -l < "$VHOST_LOG") lines)"
    else
        log_warn "No log file found"
    fi
    echo ""
}

# ============================================
# Build Functions
# ============================================

build_project() {
    log_step "Building hmCUDA project..."

    if make -C "$PROJECT_ROOT" -j"$(nproc)"; then
        log_success "Build completed successfully"
        return 0
    else
        log_error "Build failed"
        return 1
    fi
}

check_build_output() {
    local missing=0

    if [ ! -f "$VHOST_BIN" ]; then
        log_fail "Vhost binary not found: $VHOST_BIN"
        missing=1
    else
        log_success "Vhost binary exists"
    fi

    if [ ! -f "$RUNTIME_LIB" ]; then
        log_fail "Runtime library not found: $RUNTIME_LIB"
        missing=1
    else
        log_success "Runtime library exists"
    fi

    if [ ! -d "$SAMPLE_DIR" ]; then
        log_fail "Sample directory not found: $SAMPLE_DIR"
        missing=1
    else
        log_success "Sample directory exists"
    fi

    return $missing
}

# ============================================
# Utility Functions
# ============================================

ensure_sudo() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run with sudo"
        exit 1
    fi
}

wait_for_file() {
    local file="$1"
    local timeout="${2:-10}"
    local desc="${3:-File}"

    for i in $(seq 1 $timeout); do
        if [ -e "$file" ]; then
            return 0
        fi
        sleep 1
    done

    log_error "$desc not found after $timeout seconds: $file"
    return 1
}

# ============================================
# Initialization
# ============================================

# Set PROJECT_ROOT and BUILD_DIR as exports so they're available to sourcing scripts
export PROJECT_ROOT
export BUILD_DIR
export SHARE_DIR
