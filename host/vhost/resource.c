#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "resource.h"

typedef struct Resource {
    uint64_t handle;
    void *real_ptr;
    uint64_t size;
    struct Resource *next;
} Resource;

struct Context {
    uint32_t session_id;
    uint32_t vm_id;
    Resource *resources[RES_TYPE_COUNT];
    struct Context *next;
};

static Context *g_ctx_list = NULL;
static uint64_t g_next_handle = 0x10000000;

Context *hmcuda_ctx_get(uint32_t session_id, uint32_t vm_id)
{
    Context *cur = g_ctx_list;
    while (cur) {
        if (cur->session_id == session_id && cur->vm_id == vm_id) {
            return cur;
        }
        cur = cur->next;
    }
    return NULL;
}

Context *hmcuda_ctx_create(uint32_t session_id, uint32_t vm_id)
{
    Context *ctx = (Context *)calloc(1, sizeof(Context));
    ctx->session_id = session_id;
    ctx->vm_id = vm_id;
    ctx->next = g_ctx_list;
    g_ctx_list = ctx;
    return ctx;
}

static void free_resource_list(Resource *head, ResourceType type)
{
    Resource *cur = head;
    while (cur) {
        Resource *next = cur->next;
        if (cur->real_ptr) {
            if (type == RES_TYPE_MEM) {
                cudaFree(cur->real_ptr);
            } else if (type == RES_TYPE_STREAM) {
                cudaStreamDestroy((cudaStream_t)cur->real_ptr);
            } else if (type == RES_TYPE_EVENT) {
                cudaEventDestroy((cudaEvent_t)cur->real_ptr);
            } else if (type == RES_TYPE_MODULE) {
                // Modules are managed by CUDA runtime
            } else if (type == RES_TYPE_FUNCTION) {
                // Functions are managed by CUDA runtime
            } else if (type == RES_TYPE_CONTEXT) {
                // Primary contexts are managed by device lifecycle
            } else if (type == RES_TYPE_HOST_MEM) {
                cuMemFreeHost(cur->real_ptr);
            } else if (type == RES_TYPE_HOST_DEV_PTR) {
                /* CUdeviceptr is a virtual address managed by CUDA; freed via HOST_MEM cleanup */
            } else if (type == RES_TYPE_GENERIC_ALLOC) {
                cuMemRelease((CUmemGenericAllocationHandle)(uintptr_t)cur->real_ptr);
            }
        }
        free(cur);
        cur = next;
    }
}

void hmcuda_ctx_destroy(Context *ctx)
{
    if (g_ctx_list == ctx) {
        g_ctx_list = ctx->next;
    } else {
        Context *prev = g_ctx_list;
        while (prev && prev->next != ctx) {
            prev = prev->next;
        }
        if (prev) prev->next = ctx->next;
    }

    for (int i = 0; i < RES_TYPE_COUNT; i++) {
        free_resource_list(ctx->resources[i], (ResourceType)i);
    }
    free(ctx);
}

uint64_t hmcuda_res_add(Context *ctx, ResourceType type, void *real_ptr, uint64_t size)
{
    Resource *res = (Resource *)malloc(sizeof(Resource));

    if (type == RES_TYPE_MEM && size > 0) {
        // For memory: advance handle by allocation size to prevent range overlap
        // e.g., 32MB alloc at 0x10000000, next handle starts at 0x12000000
        res->handle = ++g_next_handle;
        g_next_handle += size;
    } else {
        res->handle = ++g_next_handle;
    }

    res->real_ptr = real_ptr;
    res->size = size;
    res->next = ctx->resources[type];
    ctx->resources[type] = res;
    return res->handle;
}

uint64_t hmcuda_res_add_with_handle(Context *ctx, ResourceType type, void *real_ptr, uint64_t size, uint64_t handle)
{
    Resource *res = (Resource *)malloc(sizeof(Resource));
    res->handle = handle;
    res->real_ptr = real_ptr;
    res->size = size;
    res->next = ctx->resources[type];
    ctx->resources[type] = res;
    return res->handle;
}

void *hmcuda_res_get(Context *ctx, ResourceType type, uint64_t handle)
{
    Resource *cur = ctx->resources[type];
    while (cur) {
        /* MEM and HOST_MEM both support range-based offset lookup so that
         * batched transfers can address sub-ranges by handle+byte_offset. */
        if (type == RES_TYPE_MEM || type == RES_TYPE_HOST_MEM) {
            if (handle >= cur->handle && handle < cur->handle + cur->size) {
                uint64_t offset = handle - cur->handle;
                return (char *)cur->real_ptr + offset;
            }
        } else {
            if (cur->handle == handle)
                return cur->real_ptr;
        }
        cur = cur->next;
    }
    return NULL;
}

int hmcuda_res_remove(Context *ctx, ResourceType type, uint64_t handle)
{
    Resource **pp = &ctx->resources[type];
    while (*pp) {
        Resource *cur = *pp;
        if (cur->handle == handle) {
            *pp = cur->next;
            free(cur);
            return 1;
        }
        pp = &cur->next;
    }
    return 0;
}

uint64_t hmcuda_res_get_handle(Context *ctx, ResourceType type, void *real_ptr)
{
    if (!real_ptr) return 0;

    Resource *cur = ctx->resources[type];
    while (cur) {
        if (cur->real_ptr == real_ptr) {
            return cur->handle;
        }
        cur = cur->next;
    }
    return 0;
}

int hmcuda_buffer_needs_host_ptr_patch(Context *ctx, const void *data, size_t len)
{
    Resource *hm = ctx->resources[RES_TYPE_HOST_MEM];
    if (!hm) return 0;

    for (size_t off = 0; off + 8 <= len; off += 8) {
        uint64_t v;
        memcpy(&v, (const char *)data + off, 8);
        for (Resource *r = hm; r; r = r->next) {
            if (v >= r->handle && v < r->handle + r->size)
                return 1;
        }
    }
    return 0;
}

void hmcuda_patch_embedded_host_ptrs(Context *ctx, void *data, size_t len)
{
    Resource *hm = ctx->resources[RES_TYPE_HOST_MEM];
    if (!hm) return;

    /* Scan 8-byte-aligned words in the buffer */
    for (size_t off = 0; off + 8 <= len; off += 8) {
        uint64_t v;
        memcpy(&v, (char *)data + off, 8);
        for (Resource *r = hm; r; r = r->next) {
            /* r->handle == guest_va (stored via hmcuda_res_add_with_handle) */
            if (v >= r->handle && v < r->handle + r->size) {
                uint64_t patched = (uint64_t)(uintptr_t)r->real_ptr + (v - r->handle);
                memcpy((char *)data + off, &patched, 8);
                break;
            }
        }
    }
}
