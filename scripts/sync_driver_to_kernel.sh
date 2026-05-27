#!/usr/bin/env bash
# Copy hmCUDA driver sources into an in-tree Linux kernel driver directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

KERNEL_TREE="${KERNEL_TREE:-$HOME/linux}"
DEST_DIR="${1:-$KERNEL_TREE/drivers/misc/hmcuda}"

copy_file() {
    local src="$1"
    local dst="$2"

    install -m 0644 "$PROJECT_ROOT/$src" "$DEST_DIR/$dst"
    echo "  $src -> $DEST_DIR/$dst"
}

mkdir -p "$DEST_DIR"

echo "Syncing hmCUDA driver sources to:"
echo "  $DEST_DIR"
echo ""

copy_file driver/virtio_hmcuda_main.c virtio_hmcuda_main.c
copy_file driver/cmd_runtime.c cmd_runtime.c
copy_file driver/cmd_driver.c cmd_driver.c
copy_file driver/cmd_dispatch.h cmd_dispatch.h

copy_file common/include/hmcuda_api.h hmcuda_api.h
copy_file common/include/hmcuda_types.h hmcuda_types.h
copy_file common/include/hmcuda_cmd_runtime.h hmcuda_cmd_runtime.h
copy_file common/include/hmcuda_cmd_driver.h hmcuda_cmd_driver.h
copy_file common/include/resource.h resource.h

cat > "$DEST_DIR/Makefile" <<'EOF'
obj-$(CONFIG_VIRTIO_HMCUDA) += virtio_hmcuda.o
virtio_hmcuda-y := virtio_hmcuda_main.o cmd_runtime.o cmd_driver.o
EOF

cat > "$DEST_DIR/Kconfig" <<'EOF'
config VIRTIO_HMCUDA
	tristate "VirtIO hmCUDA driver"
	depends on VIRTIO
	depends on MMU
	help
	  VirtIO guest driver for hmCUDA.
EOF

echo "  generated -> $DEST_DIR/Makefile"
echo "  generated -> $DEST_DIR/Kconfig"
echo ""
echo "Next, wire it into the kernel tree if you have not already:"
echo "  drivers/misc/Makefile: obj-\$(CONFIG_VIRTIO_HMCUDA) += hmcuda/"
echo "  drivers/misc/Kconfig:  source \"drivers/misc/hmcuda/Kconfig\""
echo ""
echo "Then enable it with:"
echo "  scripts/config --module VIRTIO_HMCUDA"
echo "  make olddefconfig"
