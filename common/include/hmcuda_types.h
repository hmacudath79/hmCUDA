/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef HMCUDA_TYPES_H
#define HMCUDA_TYPES_H

#ifdef __KERNEL__
#include <linux/types.h>
typedef u32 uint32_t;
typedef u64 uint64_t;
typedef s32 int32_t;
#else
#include <stdint.h>
#endif

/*
 * This header defines the base types and command IDs for the hmCUDA ABI.
 *
 * Pointers (e.g., to device memory) and handles (e.g., streams, events) are
 * represented as 64-bit unsigned integers (uint64_t) to ensure a consistent
 * size across different architectures.
 *
 * Command ID ranges are reserved per library so that new commands can be
 * added without renumbering:
 *   0x00001 - 0x0FFFF  CUDA Runtime API
 *   0x10000 - 0x1FFFF  CUDA Driver API
 *   0x20000 - 0x2FFFF  cuBLAS
 *   0x30000 - 0x3FFFF  cuDNN
 *   0x40000 - 0x4FFFF  cuFFT
 *   0x50000 - 0x5FFFF  cuRAND
 *   0x60000+            Reserved for future libraries
 */

/* --- Command IDs: CUDA Runtime API (0x00001 - 0x0FFFF) --- */
enum hmcuda_command_type {
    HMCUDA_CMD_MALLOC = 0x00001,
    HMCUDA_CMD_FREE,
    HMCUDA_CMD_MEMCPY,
    HMCUDA_CMD_MEMSET,
    HMCUDA_CMD_LAUNCH_KERNEL,
    HMCUDA_CMD_STREAM_CREATE,
    HMCUDA_CMD_STREAM_SYNCHRONIZE,
    HMCUDA_CMD_EVENT_CREATE,
    HMCUDA_CMD_EVENT_RECORD,
    HMCUDA_CMD_EVENT_SYNCHRONIZE,
    HMCUDA_CMD_EVENT_ELAPSED_TIME,
    HMCUDA_CMD_EVENT_DESTROY,
    HMCUDA_CMD_DEVICE_SYNCHRONIZE,
    HMCUDA_CMD_INIT,
    HMCUDA_CMD_REGISTER_FATBIN,
    HMCUDA_CMD_REGISTER_FUNCTION,
    HMCUDA_CMD_PUSH_CALL_CONFIGURATION,
    HMCUDA_CMD_POP_CALL_CONFIGURATION,
    HMCUDA_CMD_INIT_MODULE,
    HMCUDA_CMD_GET_DEVICE_PROPERTIES,
    HMCUDA_CMD_GET_DEVICE_COUNT,
    HMCUDA_CMD_SET_DEVICE,
    HMCUDA_CMD_GET_ERROR_STRING,
    HMCUDA_CMD_GET_ERROR_NAME,
    HMCUDA_CMD_RUNTIME_GET_VERSION,
    HMCUDA_CMD_FUNC_GET_ATTRIBUTES,

    /* --- Command IDs: CUDA Driver API (0x10000 - 0x1FFFF) --- */
    HMCUDA_CMD_DRIVER_BASE = 0x10000,

    /* Initialization & version */
    HMCUDA_CMD_CU_INIT = 0x10001,
    HMCUDA_CMD_CU_DRIVER_GET_VERSION,

    /* Device management */
    HMCUDA_CMD_CU_DEVICE_GET,
    HMCUDA_CMD_CU_DEVICE_GET_COUNT,
    HMCUDA_CMD_CU_DEVICE_GET_NAME,
    HMCUDA_CMD_CU_DEVICE_GET_ATTRIBUTE,
    HMCUDA_CMD_CU_DEVICE_GET_UUID,
    HMCUDA_CMD_CU_DEVICE_CAN_ACCESS_PEER,
    HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RETAIN,
    HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RELEASE,

    /* Context management */
    HMCUDA_CMD_CU_CTX_SET_CURRENT,
    HMCUDA_CMD_CU_CTX_GET_DEVICE,
    HMCUDA_CMD_CU_CTX_ENABLE_PEER_ACCESS,

    /* Stream management */
    HMCUDA_CMD_CU_STREAM_CREATE,
    HMCUDA_CMD_CU_STREAM_DESTROY,
    HMCUDA_CMD_CU_STREAM_SYNCHRONIZE,
    HMCUDA_CMD_CU_STREAM_QUERY,
    HMCUDA_CMD_CU_STREAM_WAIT_EVENT,
    HMCUDA_CMD_CU_STREAM_GET_CTX,

    /* Event management */
    HMCUDA_CMD_CU_EVENT_CREATE,
    HMCUDA_CMD_CU_EVENT_DESTROY,
    HMCUDA_CMD_CU_EVENT_RECORD,
    HMCUDA_CMD_CU_EVENT_ELAPSED_TIME,

    /* Memory allocation */
    HMCUDA_CMD_CU_MEM_ALLOC,
    HMCUDA_CMD_CU_MEM_FREE,
    HMCUDA_CMD_CU_MEM_HOST_ALLOC,
    HMCUDA_CMD_CU_MEM_FREE_HOST,
    HMCUDA_CMD_CU_MEMSET_D32,

    /* Memory copy */
    HMCUDA_CMD_CU_MEMCPY_ASYNC,
    HMCUDA_CMD_CU_MEMCPY_HTOD,
    HMCUDA_CMD_CU_MEMCPY, /* H2D or D2H via page-pinning virtqueue path */

    /* Pointer & error */
    HMCUDA_CMD_CU_POINTER_GET_ATTRIBUTE,
    HMCUDA_CMD_CU_GET_ERROR_STRING,
    HMCUDA_CMD_CU_GET_ERROR_NAME,

    /* Virtual memory management */
    HMCUDA_CMD_CU_MEM_CREATE,
    HMCUDA_CMD_CU_MEM_RELEASE,
    HMCUDA_CMD_CU_MEM_MAP,
    HMCUDA_CMD_CU_MEM_UNMAP,
    HMCUDA_CMD_CU_MEM_ADDRESS_RESERVE,
    HMCUDA_CMD_CU_MEM_ADDRESS_FREE,
    HMCUDA_CMD_CU_MEM_EXPORT_TO_SHAREABLE_HANDLE,
    HMCUDA_CMD_CU_MEM_IMPORT_FROM_SHAREABLE_HANDLE,
    HMCUDA_CMD_CU_MEM_SET_ACCESS,
    HMCUDA_CMD_CU_MEM_GET_ALLOC_GRANULARITY,

    /* Multicast memory */
    HMCUDA_CMD_CU_MULTICAST_CREATE,
    HMCUDA_CMD_CU_MULTICAST_GET_GRANULARITY,
    HMCUDA_CMD_CU_MULTICAST_ADD_DEVICE,
    HMCUDA_CMD_CU_MULTICAST_BIND_MEM,
    HMCUDA_CMD_CU_MULTICAST_UNBIND,

    /* --- Command IDs: cuBLAS (0x20000 - 0x2FFFF) --- */
    HMCUDA_CMD_CUBLAS_BASE = 0x20000,
};

/* Common header for all requests */
struct hmcuda_req_header {
    uint32_t cmd_type;
    uint32_t flags; /* For versioning or other metadata */
};

/* Common header for all responses */
struct hmcuda_resp_header {
    uint32_t cmd_type;
    uint32_t cuda_error; /* Corresponds to cudaError_t */
};

/* C-struct equivalent for dim3 */
struct hmcuda_dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

#endif /* HMCUDA_TYPES_H */
