/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef HMCUDA_DRV_CMD_DISPATCH_H
#define HMCUDA_DRV_CMD_DISPATCH_H

#include <linux/types.h>

/*
 * Per-library command dispatch interface for the guest kernel driver.
 *
 * Each library (runtime, driver API, cuBLAS, ...) implements
 * hmcuda_<lib>_get_sizes().  The main ioctl handler calls
 * hmcuda_cmd_get_sizes() which tries each library in turn.
 *
 * To add a new library:
 *   1. Create cmd_<lib>.c implementing hmcuda_<lib>_get_sizes()
 *   2. Add the call in hmcuda_cmd_get_sizes() (virtio_hmcuda_main.c)
 *   3. Add the .o to Kbuild
 */

struct hmcuda_cmd_sizes {
    size_t req_size;   /* Total request payload size (fixed + variable, excluding header) */
    size_t resp_size;  /* Fixed response payload size (excluding header) */
};

/*
 * Each library's get_sizes function:
 *   - Returns 0 and fills *sizes if the command belongs to this library.
 *   - Returns -EINVAL if the command is not recognized.
 *   - req_payload may be NULL if req_size is 0 (no fixed struct to read).
 *     When non-NULL, it points to the fixed request struct already copied
 *     from userspace, so the function can compute variable-length extras.
 */

/* CUDA Runtime API */
int hmcuda_runtime_get_sizes(uint32_t cmd, const void *req_payload,
                             struct hmcuda_cmd_sizes *sizes);

/* CUDA Driver API */
int hmcuda_driver_get_sizes(uint32_t cmd, const void *req_payload,
                            struct hmcuda_cmd_sizes *sizes);

/* Future libraries:
 * int hmcuda_cublas_get_sizes(uint32_t cmd, const void *req_payload,
 *                             struct hmcuda_cmd_sizes *sizes);
 */

#endif /* HMCUDA_DRV_CMD_DISPATCH_H */
