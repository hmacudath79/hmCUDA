#ifndef RESOURCE_H
#define RESOURCE_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    RES_TYPE_MEM,
    RES_TYPE_STREAM,
    RES_TYPE_EVENT,
    RES_TYPE_MODULE,
    RES_TYPE_FUNCTION,
    RES_TYPE_CONTEXT,
    RES_TYPE_HOST_MEM,
    RES_TYPE_HOST_DEV_PTR, /* CUdeviceptr from cuMemHostGetDevicePointer, keyed by guest_va */
    RES_TYPE_GENERIC_ALLOC,
    RES_TYPE_COUNT
} ResourceType;

typedef struct Context Context;

Context *hmcuda_ctx_create(uint32_t session_id, uint32_t vm_id);
Context *hmcuda_ctx_get(uint32_t session_id, uint32_t vm_id);
void hmcuda_ctx_destroy(Context *ctx);

uint64_t hmcuda_res_add(Context *ctx, ResourceType type, void *real_ptr, uint64_t size);
uint64_t hmcuda_res_add_with_handle(Context *ctx, ResourceType type, void *real_ptr, uint64_t size, uint64_t handle);
void *hmcuda_res_get(Context *ctx, ResourceType type, uint64_t handle);
uint64_t hmcuda_res_get_handle(Context *ctx, ResourceType type, void *real_ptr);
int hmcuda_res_remove(Context *ctx, ResourceType type, uint64_t handle);

/*
 * Scan an H2D data buffer for 8-byte-aligned values that fall within any
 * RES_TYPE_HOST_MEM allocation range (guest_va..guest_va+size) and replace
 * them with the corresponding backend real_ptr offset.  This translates
 * linked-list next-pointers written by the guest CPU into guest_va space so
 * the GPU can dereference them after the data lands in device or pinned memory.
 */
void hmcuda_patch_embedded_host_ptrs(Context *ctx, void *data, size_t len);
int hmcuda_buffer_needs_host_ptr_patch(Context *ctx, const void *data, size_t len);

#endif
