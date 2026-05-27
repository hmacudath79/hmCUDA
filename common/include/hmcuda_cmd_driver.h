/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef HMCUDA_CMD_DRIVER_H
#define HMCUDA_CMD_DRIVER_H

#include "hmcuda_types.h"

/*
 * CUDA Driver API request/response structures.
 *
 * Each pair corresponds to a HMCUDA_CMD_CU_* command from hmcuda_types.h.
 * Structures without a _resp counterpart return only the common
 * hmcuda_resp_header (i.e. just the CUresult code).
 *
 * Driver API handles (CUcontext, CUstream, CUevent, CUdeviceptr, etc.)
 * are transported as uint64_t for ABI stability.
 */

/* ------------------------------------------------------------------ */
/*  Initialization & version                                          */
/* ------------------------------------------------------------------ */

/* cuInit */
struct hmcuda_cu_init_req {
    uint32_t flags;
};

/* cuDriverGetVersion */
struct hmcuda_cu_driver_get_version_resp {
    int32_t version;
};

/* ------------------------------------------------------------------ */
/*  Device management                                                 */
/* ------------------------------------------------------------------ */

/* cuDeviceGet */
struct hmcuda_cu_device_get_req {
    int32_t ordinal;
};
struct hmcuda_cu_device_get_resp {
    int32_t device; /* CUdevice */
};

/* cuDeviceGetCount */
struct hmcuda_cu_device_get_count_resp {
    int32_t count;
};

/* cuDeviceGetName */
struct hmcuda_cu_device_get_name_req {
    int32_t device;
    int32_t name_len; /* max bytes to return (including NUL) */
};
struct hmcuda_cu_device_get_name_resp {
    int32_t name_len;
    /* name string follows (name_len bytes, NUL-terminated) */
};

/* cuDeviceGetAttribute */
struct hmcuda_cu_device_get_attribute_req {
    int32_t attrib; /* CUdevice_attribute */
    int32_t device;
};
struct hmcuda_cu_device_get_attribute_resp {
    int32_t value;
};

/* cuDeviceGetUuid */
struct hmcuda_cu_device_get_uuid_req {
    int32_t device;
};
struct hmcuda_cu_device_get_uuid_resp {
    uint8_t uuid[16]; /* CUuuid bytes */
};

/* cuDeviceCanAccessPeer */
struct hmcuda_cu_device_can_access_peer_req {
    int32_t device;
    int32_t peer_device;
};
struct hmcuda_cu_device_can_access_peer_resp {
    int32_t can_access; /* 0 or 1 */
};

/* cuDevicePrimaryCtxRetain */
struct hmcuda_cu_device_primary_ctx_retain_req {
    int32_t device;
};
struct hmcuda_cu_device_primary_ctx_retain_resp {
    uint64_t context; /* CUcontext */
};

/* cuDevicePrimaryCtxRelease */
struct hmcuda_cu_device_primary_ctx_release_req {
    int32_t device;
};

/* ------------------------------------------------------------------ */
/*  Context management                                                */
/* ------------------------------------------------------------------ */

/* cuCtxSetCurrent */
struct hmcuda_cu_ctx_set_current_req {
    uint64_t context;
};

/* cuCtxGetDevice */
struct hmcuda_cu_ctx_get_device_resp {
    int32_t device;
};

/* cuCtxEnablePeerAccess */
struct hmcuda_cu_ctx_enable_peer_access_req {
    uint64_t peer_context;
    uint32_t flags;
};

/* ------------------------------------------------------------------ */
/*  Stream management                                                 */
/* ------------------------------------------------------------------ */

/* cuStreamCreate */
struct hmcuda_cu_stream_create_req {
    uint32_t flags;
};
struct hmcuda_cu_stream_create_resp {
    uint64_t stream;
};

/* cuStreamDestroy */
struct hmcuda_cu_stream_destroy_req {
    uint64_t stream;
};

/* cuStreamSynchronize */
struct hmcuda_cu_stream_synchronize_req {
    uint64_t stream;
};

/* cuStreamQuery */
struct hmcuda_cu_stream_query_req {
    uint64_t stream;
};

/* cuStreamWaitEvent */
struct hmcuda_cu_stream_wait_event_req {
    uint64_t stream;
    uint64_t event;
    uint32_t flags;
};

/* cuStreamGetCtx */
struct hmcuda_cu_stream_get_ctx_req {
    uint64_t stream;
};
struct hmcuda_cu_stream_get_ctx_resp {
    uint64_t context;
};

/* ------------------------------------------------------------------ */
/*  Event management                                                  */
/* ------------------------------------------------------------------ */

/* cuEventCreate */
struct hmcuda_cu_event_create_req {
    uint32_t flags;
};
struct hmcuda_cu_event_create_resp {
    uint64_t event;
};

/* cuEventDestroy */
struct hmcuda_cu_event_destroy_req {
    uint64_t event;
};

/* cuEventRecord */
struct hmcuda_cu_event_record_req {
    uint64_t event;
    uint64_t stream;
};

/* cuEventElapsedTime */
struct hmcuda_cu_event_elapsed_time_req {
    uint64_t start_event;
    uint64_t end_event;
};
struct hmcuda_cu_event_elapsed_time_resp {
    float ms;
};

/* ------------------------------------------------------------------ */
/*  Memory allocation                                                 */
/* ------------------------------------------------------------------ */

/* cuMemAlloc */
struct hmcuda_cu_mem_alloc_req {
    uint64_t bytesize;
};
struct hmcuda_cu_mem_alloc_resp {
    uint64_t dptr; /* CUdeviceptr */
};

/* cuMemFree */
struct hmcuda_cu_mem_free_req {
    uint64_t dptr;
};

/* cuMemHostAlloc */
struct hmcuda_cu_mem_host_alloc_req {
    uint64_t bytesize;
    uint32_t flags; /* CU_MEMHOSTALLOC_PORTABLE etc. */
    uint64_t guest_va; /* guest malloc'd VA returned to app; used as backend map key */
    uint64_t byte_offset; /* offset into real_ptr for this batch (0 on first batch) */
};
/* No response struct — no handle returned; guest already has guest_va */

/* cuMemFreeHost */
struct hmcuda_cu_mem_free_host_req {
    uint64_t guest_va; /* key used when the allocation was registered */
};

/* cuMemsetD32 */
struct hmcuda_cu_memset_d32_req {
    uint64_t dptr;
    uint32_t value;
    uint64_t count; /* number of uint32_t elements */
};

/* ------------------------------------------------------------------ */
/*  Memory copy                                                       */
/* ------------------------------------------------------------------ */

/* cuMemcpyAsync */
struct hmcuda_cu_memcpy_async_req {
    uint64_t dst;
    uint64_t src;
    uint64_t bytesize;
    uint64_t stream;
};

/* cuMemcpyHtoD */
struct hmcuda_cu_memcpy_htod_req {
    uint64_t dst_device;
    uint64_t src_host; /* guest user-space address (for data transfer) */
    uint64_t bytesize;
};

/* ------------------------------------------------------------------ */
/*  Pointer & error                                                   */
/* ------------------------------------------------------------------ */

/* cuPointerGetAttribute */
struct hmcuda_cu_pointer_get_attribute_req {
    uint32_t attrib; /* CUpointer_attribute */
    uint64_t ptr;
};
struct hmcuda_cu_pointer_get_attribute_resp {
    uint64_t value; /* widened to hold any attribute result */
};

/* cuGetErrorString */
struct hmcuda_cu_get_error_string_req {
    uint32_t error; /* CUresult */
};
struct hmcuda_cu_get_error_string_resp {
    uint32_t str_len;
    /* string follows (str_len bytes, NUL-terminated) */
};

/* cuGetErrorName */
struct hmcuda_cu_get_error_name_req {
    uint32_t error;
};
struct hmcuda_cu_get_error_name_resp {
    uint32_t str_len;
    /* string follows (str_len bytes, NUL-terminated) */
};

/* ------------------------------------------------------------------ */
/*  Virtual memory management                                         */
/* ------------------------------------------------------------------ */

/* Serialized CUmemAllocationProp (subset used by nvbandwidth) */
struct hmcuda_cu_mem_alloc_prop {
    uint32_t type;                  /* CUmemAllocationType */
    uint32_t location_type;         /* CUmemLocationType */
    int32_t  location_id;           /* device ordinal */
    uint32_t requested_handle_types; /* CUmemAllocationHandleType */
};

/* cuMemCreate */
struct hmcuda_cu_mem_create_req {
    uint64_t size;
    struct hmcuda_cu_mem_alloc_prop prop;
    uint64_t flags;
};
struct hmcuda_cu_mem_create_resp {
    uint64_t handle; /* CUmemGenericAllocationHandle */
};

/* cuMemRelease */
struct hmcuda_cu_mem_release_req {
    uint64_t handle;
};

/* cuMemMap */
struct hmcuda_cu_mem_map_req {
    uint64_t ptr;    /* virtual address */
    uint64_t size;
    uint64_t offset;
    uint64_t handle;
    uint64_t flags;
};

/* cuMemUnmap */
struct hmcuda_cu_mem_unmap_req {
    uint64_t ptr;
    uint64_t size;
};

/* cuMemAddressReserve */
struct hmcuda_cu_mem_address_reserve_req {
    uint64_t size;
    uint64_t alignment;
    uint64_t addr;  /* base VA hint */
    uint64_t flags;
};
struct hmcuda_cu_mem_address_reserve_resp {
    uint64_t ptr;
};

/* cuMemAddressFree */
struct hmcuda_cu_mem_address_free_req {
    uint64_t ptr;
    uint64_t size;
};

/* Serialized CUmemAccessDesc */
struct hmcuda_cu_mem_access_desc {
    uint32_t location_type;
    int32_t  location_id;
    uint32_t flags; /* CUmemAccess_flags */
};

/* cuMemSetAccess */
struct hmcuda_cu_mem_set_access_req {
    uint64_t ptr;
    uint64_t size;
    uint32_t count; /* number of hmcuda_cu_mem_access_desc that follow */
    /* count × struct hmcuda_cu_mem_access_desc follows */
};

/* cuMemGetAllocationGranularity */
struct hmcuda_cu_mem_get_alloc_granularity_req {
    struct hmcuda_cu_mem_alloc_prop prop;
    uint32_t option; /* CUmemAllocationGranularity_flags */
};
struct hmcuda_cu_mem_get_alloc_granularity_resp {
    uint64_t granularity;
};

/* cuMemExportToShareableHandle */
struct hmcuda_cu_mem_export_req {
    uint64_t handle;
    uint32_t handle_type; /* CUmemAllocationHandleType */
    uint64_t flags;
};
struct hmcuda_cu_mem_export_resp {
    uint8_t shareable_handle[64]; /* large enough for CUmemFabricHandle */
};

/* cuMemImportFromShareableHandle */
struct hmcuda_cu_mem_import_req {
    uint8_t shareable_handle[64];
    uint32_t handle_type;
};
struct hmcuda_cu_mem_import_resp {
    uint64_t handle;
};

/* ------------------------------------------------------------------ */
/*  Multicast memory                                                  */
/* ------------------------------------------------------------------ */

/* Serialized CUmulticastObjectProp */
struct hmcuda_cu_multicast_prop {
    uint32_t num_devices;
    uint32_t handle_types;
    uint64_t size;
};

/* cuMulticastCreate */
struct hmcuda_cu_multicast_create_req {
    struct hmcuda_cu_multicast_prop prop;
};
struct hmcuda_cu_multicast_create_resp {
    uint64_t mc_handle;
};

/* cuMulticastGetGranularity */
struct hmcuda_cu_multicast_get_granularity_req {
    struct hmcuda_cu_multicast_prop prop;
    uint32_t option; /* CUmulticastGranularity_flags */
};
struct hmcuda_cu_multicast_get_granularity_resp {
    uint64_t granularity;
};

/* cuMulticastAddDevice */
struct hmcuda_cu_multicast_add_device_req {
    uint64_t mc_handle;
    int32_t device;
};

/* cuMulticastBindMem */
struct hmcuda_cu_multicast_bind_mem_req {
    uint64_t mc_handle;
    uint64_t mc_offset;
    uint64_t mem_handle;
    uint64_t mem_offset;
    uint64_t size;
    uint64_t flags;
};

/* cuMulticastUnbind */
struct hmcuda_cu_multicast_unbind_req {
    uint64_t mc_handle;
    int32_t device;
    uint64_t mc_offset;
    uint64_t size;
};

#endif /* HMCUDA_CMD_DRIVER_H */
