#!/usr/bin/env bash
# Build a tiny initramfs containing busybox, hmCUDA guest libs, and CUDA tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
INITRAMFS_ROOT="${INITRAMFS_ROOT:-$BUILD_DIR/initramfs-root}"
INITRAMFS_OUT="${INITRAMFS_OUT:-$BUILD_DIR/hmcuda-initramfs.cpio.gz}"
MANIFEST_OUT="${MANIFEST_OUT:-$BUILD_DIR/hmcuda-initramfs.manifest}"
BUSYBOX="${BUSYBOX:-$(command -v busybox || true)}"
INIT_TESTS="${INIT_TESTS:-smallTest matrixMul}"
AUTO_TEST="${AUTO_TEST:-0}"
INCLUDE_HOST_CUDA_LIBS="${INCLUDE_HOST_CUDA_LIBS:-0}"
PRINT_MANIFEST="${PRINT_MANIFEST:-1}"

copy_to_root() {
    local src="$1"
    local dst="$2"

    if [ ! -e "$src" ]; then
        echo "missing: $src" >&2
        return 1
    fi

    mkdir -p "$INITRAMFS_ROOT/$(dirname "$dst")"
    cp -aL "$src" "$INITRAMFS_ROOT/$dst"
}

copy_elf_runtime_deps() {
    local elf="$1"
    local interp dep dep_base

    if ! file "$elf" | grep -q 'ELF'; then
        return 0
    fi

    interp="$(readelf -l "$elf" 2>/dev/null | sed -n 's/.*Requesting program interpreter: \(.*\)]/\1/p' | head -1)"
    if [ -n "$interp" ] && [ -e "$interp" ]; then
        copy_to_root "$interp" "$interp"
    fi

    while read -r dep; do
        [ -n "$dep" ] || continue
        [ -e "$dep" ] || continue
        dep_base="$(basename "$dep")"
        if [ "$INCLUDE_HOST_CUDA_LIBS" != "1" ]; then
            case "$dep_base" in
                libcudart.so*|libcuda.so*|libnvidia-ml.so*)
                    continue
                    ;;
            esac
        fi
        copy_to_root "$dep" "$dep"
    done < <(
        ldd "$elf" 2>/dev/null |
            awk '
                $2 == "=>" && $3 ~ /^\// { print $3 }
                $1 ~ /^\// { print $1 }
            ' |
            sort -u
    )
}

if [ -z "$BUSYBOX" ] || [ ! -x "$BUSYBOX" ]; then
    echo "busybox not found. Install it or set BUSYBOX=/path/to/busybox." >&2
    exit 1
fi

for tool in file readelf ldd cpio gzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "required tool not found: $tool" >&2
        exit 1
    fi
done

rm -rf "$INITRAMFS_ROOT"
mkdir -p "$INITRAMFS_ROOT"/{bin,sbin,etc,proc,sys,dev,tmp,run,lib,lib64,usr/bin,usr/lib,usr/lib64,tests,lib/hmcuda}
chmod 0755 "$INITRAMFS_ROOT"
chmod 1777 "$INITRAMFS_ROOT/tmp"

copy_to_root "$BUSYBOX" /bin/busybox
copy_elf_runtime_deps "$BUSYBOX"

for applet in sh mount umount mkdir mknod modprobe sleep echo cat ls dmesg uname env true false grep find head tail readlink; do
    ln -sf busybox "$INITRAMFS_ROOT/bin/$applet"
done
ln -sf ../bin/busybox "$INITRAMFS_ROOT/sbin/modprobe"

if compgen -G "$BUILD_DIR/guest/intercept/lib*.so*" >/dev/null; then
    cp -aL "$BUILD_DIR"/guest/intercept/lib*.so* "$INITRAMFS_ROOT/lib/hmcuda/"
    while IFS= read -r lib; do
        copy_elf_runtime_deps "$lib"
    done < <(find "$BUILD_DIR/guest/intercept" -maxdepth 1 -type f -name 'lib*.so*' | sort)
else
    echo "warning: no hmCUDA guest intercept libraries found under $BUILD_DIR/guest/intercept" >&2
fi

sample_count=0
if [ -d "$BUILD_DIR/sample" ]; then
    while IFS= read -r sample; do
        copy_to_root "$sample" "/tests/$(basename "$sample")"
        copy_elf_runtime_deps "$sample"
        sample_count=$((sample_count + 1))
    done < <(find "$BUILD_DIR/sample" -maxdepth 1 -type f -perm /111 | sort)
fi

if [ -x "$BUILD_DIR/third_party/nvbandwidth/nvbandwidth" ]; then
    copy_to_root "$BUILD_DIR/third_party/nvbandwidth/nvbandwidth" /tests/nvbandwidth
    copy_elf_runtime_deps "$BUILD_DIR/third_party/nvbandwidth/nvbandwidth"
fi

cat > "$INITRAMFS_ROOT/init" <<EOF
#!/bin/sh
export PATH=/bin:/sbin:/usr/bin:/usr/sbin
export LD_LIBRARY_PATH=/lib/hmcuda:/lib:/lib64:/usr/lib:/usr/lib64
export HMCUDA_TRANSPORT=\${HMCUDA_TRANSPORT:-virtio}
export HMCUDA_DEBUG=\${HMCUDA_DEBUG:-2}

mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true

echo "hmCUDA initramfs booted"
echo "kernel: \$(uname -a)"

echo ""
echo "Diagnostics:"
echo "  /check_hmcuda"
echo "Manual tests:"
echo "  HMCUDA_DEBUG=4 /tests/smallTest"
echo "  HMCUDA_DEBUG=4 /tests/matrixMul"
echo ""

if [ "$AUTO_TEST" = "1" ]; then
    failed=0
    for test in $INIT_TESTS; do
        if [ -x "/tests/\$test" ]; then
            echo "running /tests/\$test"
            "/tests/\$test" || failed=1
        else
            echo "skip missing test: \$test"
        fi
    done
    echo "hmCUDA tests complete: failed=\$failed"
fi

exec /bin/sh
EOF
chmod 0755 "$INITRAMFS_ROOT/init"

cat > "$INITRAMFS_ROOT/check_hmcuda" <<'EOF'
#!/bin/sh
export PATH=/bin:/sbin:/usr/bin:/usr/sbin

mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true

echo "== kernel =="
uname -a

echo ""
echo "== /dev =="
ls -l /dev/virtio-hmcuda 2>/dev/null || echo "missing: /dev/virtio-hmcuda"

echo ""
echo "== misc device registration =="
grep -i virtio-hmcuda /proc/misc 2>/dev/null || echo "no virtio-hmcuda entry in /proc/misc"
if [ -d /sys/class/misc/virtio-hmcuda ]; then
    printf 'sysfs: '
    readlink /sys/class/misc/virtio-hmcuda 2>/dev/null || echo "/sys/class/misc/virtio-hmcuda"
else
    echo "missing: /sys/class/misc/virtio-hmcuda"
fi

echo ""
echo "== virtio bus =="
if [ -d /sys/bus/virtio/devices ]; then
    for d in /sys/bus/virtio/devices/*; do
        [ -e "$d" ] || continue
        printf '%s device=' "$d"
        cat "$d/device" 2>/dev/null || true
        printf '%s driver=' "$d"
        readlink "$d/driver" 2>/dev/null || echo "none"
    done
else
    echo "missing: /sys/bus/virtio/devices"
fi

echo ""
echo "== pci devices =="
if [ -d /sys/bus/pci/devices ]; then
    for d in /sys/bus/pci/devices/*; do
        [ -e "$d/vendor" ] || continue
        vendor="$(cat "$d/vendor" 2>/dev/null)"
        device="$(cat "$d/device" 2>/dev/null)"
        case "$vendor:$device" in
            0x1af4:*) echo "$d vendor=$vendor device=$device" ;;
        esac
    done
else
    echo "missing: /sys/bus/pci/devices"
fi

echo ""
echo "== hmCUDA libs/tests =="
ls -l /lib/hmcuda 2>/dev/null || true
ls -l /tests 2>/dev/null || true

echo ""
echo "== recent kernel logs =="
dmesg | grep -i -E 'virtio|hmcuda|vhost|pci' | tail -80
EOF
chmod 0755 "$INITRAMFS_ROOT/check_hmcuda"

mkdir -p "$(dirname "$INITRAMFS_OUT")"
mkdir -p "$(dirname "$MANIFEST_OUT")"
(
    cd "$INITRAMFS_ROOT"
    find . -mindepth 1 -printf '%M %9s %p\n' | sort -k3
) > "$MANIFEST_OUT"

(
    cd "$INITRAMFS_ROOT"
    find . -print0 | cpio --null -ov --format=newc 2>/dev/null | gzip -9
) > "$INITRAMFS_OUT"

echo "Packed initramfs: $INITRAMFS_OUT"
echo "Root staging dir: $INITRAMFS_ROOT"
echo "Busybox: $BUSYBOX"
echo "Guest sample binaries: $sample_count"
echo "Host CUDA/NVML libs included: $INCLUDE_HOST_CUDA_LIBS"
echo "Manifest: $MANIFEST_OUT"

if [ "$PRINT_MANIFEST" = "1" ]; then
    echo ""
    echo "Packed files:"
    cat "$MANIFEST_OUT"
fi
