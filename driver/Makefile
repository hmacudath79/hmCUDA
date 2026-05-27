# SPDX-License-Identifier: GPL-2.0-only
all:
	@echo "virtio_hmcuda is built in-tree by the external Linux kernel project." >&2
	@echo "Do not build it as an out-of-tree module from this repository." >&2
	@exit 1

clean:
	@echo "Nothing to clean: driver build artifacts live in the external Linux kernel project."

install:
	@echo "Install virtio_hmcuda from the external Linux kernel project." >&2
	@exit 1

load:
	sudo modprobe virtio_hmcuda

unload:
	sudo rmmod virtio_hmcuda || true

reload: unload load

.PHONY: all clean install load unload reload
