#!/usr/bin/env bash
# Select the CUDA toolkit and nvcc host compiler used by hmCUDA builds.
# Usage: source scripts/cuda_env.sh [cuda-root|version] [gcc-root|gcc-version|gcc-bin]

_hmcuda_cuda_env_sourced=0
if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
    _hmcuda_cuda_env_sourced=1
fi

_hmcuda_cuda_env_usage() {
    cat <<'EOF'
Usage: source scripts/cuda_env.sh [cuda-root|version] [gcc-root|gcc-version|gcc-bin]

Examples:
  source scripts/cuda_env.sh                  # use /usr/local/cuda and auto-detect GCC
  source scripts/cuda_env.sh 13.2 13          # use /usr/local/cuda-13.2 with g++-13
  source scripts/cuda_env.sh 13.2 /opt/gcc-13 # use GCC installed under /opt/gcc-13
  source scripts/cuda_env.sh /opt/cuda g++-13 # use explicit toolkit path and compiler

Variables exported:
  CUDA_HOME, CUDA_PATH, CUDACXX, CUDA_HOST_COMPILER, CC, CXX, PATH, LD_LIBRARY_PATH
EOF
}

if [ "$_hmcuda_cuda_env_sourced" -ne 1 ]; then
    _hmcuda_cuda_env_usage
    echo ""
    echo "This script must be sourced so it can update your shell environment."
    exit 1
fi

_hmcuda_cuda_arg="${1:-}"
_hmcuda_gcc_arg="${2:-13}"

if [ "$_hmcuda_cuda_arg" = "-h" ] || [ "$_hmcuda_cuda_arg" = "--help" ]; then
    _hmcuda_cuda_env_usage
    return 0
fi

if [ -z "$_hmcuda_cuda_arg" ]; then
    if [ -d /usr/local/cuda ]; then
        _hmcuda_cuda_root=/usr/local/cuda
    else
        _hmcuda_cuda_root="$(find /usr/local -maxdepth 1 -type d -name 'cuda*' 2>/dev/null | sort -Vr | head -1)"
    fi
elif [ -d "$_hmcuda_cuda_arg" ]; then
    _hmcuda_cuda_root="$(cd "$_hmcuda_cuda_arg" && pwd)"
elif [ -d "/usr/local/cuda-$_hmcuda_cuda_arg" ]; then
    _hmcuda_cuda_root="/usr/local/cuda-$_hmcuda_cuda_arg"
else
    echo "CUDA toolkit not found for '$_hmcuda_cuda_arg'" >&2
    return 1
fi

if [ ! -x "$_hmcuda_cuda_root/bin/nvcc" ]; then
    echo "nvcc not found at $_hmcuda_cuda_root/bin/nvcc" >&2
    return 1
fi

_hmcuda_resolve_gxx() {
    local arg="$1"

    if [ -n "$arg" ]; then
        if [ -d "$arg" ]; then
            if [ -x "$arg/bin/g++" ]; then
                printf '%s/bin/g++' "$arg"
                return 0
            fi
            if [ -x "$arg/bin/g++-13" ]; then
                printf '%s/bin/g++-13' "$arg"
                return 0
            fi
        elif [ -x "$arg" ]; then
            printf '%s' "$arg"
            return 0
        elif command -v "$arg" >/dev/null 2>&1; then
            command -v "$arg"
            return 0
        elif command -v "g++-$arg" >/dev/null 2>&1; then
            command -v "g++-$arg"
            return 0
        fi

        return 1
    fi

    if command -v g++-13 >/dev/null 2>&1; then
        command -v g++-13
        return 0
    fi

    if command -v g++ >/dev/null 2>&1; then
        command -v g++
        return 0
    fi

    return 1
}

_hmcuda_gxx="$(_hmcuda_resolve_gxx "$_hmcuda_gcc_arg")"
if [ -z "$_hmcuda_gxx" ]; then
    echo "Requested g++ not found: $_hmcuda_gcc_arg" >&2
    echo "Install GCC 13 or pass its path as the second argument, for example:" >&2
    echo "  source scripts/cuda_env.sh 13.2 /opt/gcc-13" >&2
    return 1
fi

_hmcuda_gcc="${_hmcuda_gxx%/*}/gcc${_hmcuda_gxx##*g++}"
if [ ! -x "$_hmcuda_gcc" ]; then
    _hmcuda_gcc="${_hmcuda_gxx%/*}/gcc"
fi
if [ ! -x "$_hmcuda_gcc" ]; then
    _hmcuda_gcc="$(command -v gcc || true)"
fi

_hmcuda_path_prepend() {
    if [ -z "$2" ]; then
        printf '%s' "$1"
        return
    fi
    case ":$2:" in
        *":$1:"*) printf '%s' "$2" ;;
        *) printf '%s:%s' "$1" "$2" ;;
    esac
}

export CUDA_HOME="$_hmcuda_cuda_root"
export CUDA_PATH="$_hmcuda_cuda_root"
export CUDACXX="$_hmcuda_cuda_root/bin/nvcc"
export CUDA_HOST_COMPILER="$_hmcuda_gxx"
export CXX="$_hmcuda_gxx"
export CC="$_hmcuda_gcc"
export PATH="$(_hmcuda_path_prepend "$_hmcuda_cuda_root/bin" "${PATH:-}")"
export PATH="$(_hmcuda_path_prepend "$(dirname "$_hmcuda_gxx")" "$PATH")"

if [ -d "$_hmcuda_cuda_root/lib64" ]; then
    export LD_LIBRARY_PATH="$(_hmcuda_path_prepend "$_hmcuda_cuda_root/lib64" "${LD_LIBRARY_PATH:-}")"
elif [ -d "$_hmcuda_cuda_root/lib" ]; then
    export LD_LIBRARY_PATH="$(_hmcuda_path_prepend "$_hmcuda_cuda_root/lib" "${LD_LIBRARY_PATH:-}")"
fi

echo "hmCUDA CUDA environment:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  nvcc=$($CUDACXX --version | sed -n 's/^.*release \([^,]*\).*$/\1/p' | head -1)"
echo "  CUDA_HOST_COMPILER=$CUDA_HOST_COMPILER"
echo "  gcc=$($CC -dumpfullversion -dumpversion 2>/dev/null || true)"

unset _hmcuda_cuda_arg _hmcuda_cuda_root _hmcuda_cuda_env_sourced _hmcuda_gcc_arg _hmcuda_gxx _hmcuda_gcc
unset -f _hmcuda_cuda_env_usage _hmcuda_path_prepend _hmcuda_resolve_gxx
