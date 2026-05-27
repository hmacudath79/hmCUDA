#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "hmcuda_api.h"
#include "resource.h"
#include "log.h"
#include "libvhost-user.h"

extern VuDev *hmcuda_vu_dev_get(void);

static bool g_guest_mem_registered = false;

static void hmcuda_register_guest_memory(void)
{
    VuDev *dev = hmcuda_vu_dev_get();
    if (!dev || g_guest_mem_registered) return;

    for (uint32_t i = 0; i < dev->nregions; i++) {
        VuDevRegion *r = &dev->regions[i];
        void *ptr  = (void *)(uintptr_t)(r->mmap_addr + r->mmap_offset);
        size_t sz  = (size_t)r->size;
        if (!ptr || !sz) continue;
        CUresult res = cuMemHostRegister(ptr, sz, CU_MEMHOSTREGISTER_PORTABLE);
        LOG("cuMemHostRegister: region %u ptr=%p size=%zu -> %d", i, ptr, sz, res);
    }
    g_guest_mem_registered = true;
}

/*
 * Resolve a guest handle to a real pointer, trying device memory first,
 * then host pinned memory.  Used by cuMemcpyAsync where either type is valid.
 */
static void *resolve_any_mem(Context *ctx, uint64_t handle)
{
    void *ptr = hmcuda_res_get(ctx, RES_TYPE_MEM, handle);
    if (ptr) return ptr;
    return hmcuda_res_get(ctx, RES_TYPE_HOST_MEM, handle);
}

static CUstream resolve_stream(Context *ctx, uint64_t stream_handle)
{
    if (stream_handle == 0)
        return NULL;
    if (stream_handle == (uint64_t)(uintptr_t)CU_STREAM_LEGACY)
        return CU_STREAM_LEGACY;
    if (stream_handle == (uint64_t)(uintptr_t)CU_STREAM_PER_THREAD)
        return CU_STREAM_PER_THREAD;
    CUstream s = (CUstream)hmcuda_res_get(ctx, RES_TYPE_STREAM, stream_handle);
    return s ? s : NULL;
}

/* ================================================================== */
/*  Context management                                                */
/* ================================================================== */

uint32_t hmcuda_core_cu_device_primary_ctx_retain(uint32_t sid, uint32_t vid,
                                                   int32_t device, uint64_t *ctx_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUcontext pctx;
    CUresult res = cuDevicePrimaryCtxRetain(&pctx, (CUdevice)device);
    if (res != CUDA_SUCCESS) {
        LOG("hmcuda_core_cu_device_primary_ctx_retain: device=%d, error=%d; attempting device reset",
            device, res);
        /* Primary context may be in a sticky error state (e.g. error 700 from a
         * previous kernel trap).  Reset the device, purge the stale resource
         * table, and retry once. */
        cudaDeviceReset();
        /* All handles (streams, events, modules, functions, memory) are now
         * invalid — recreate a clean context table. */
        hmcuda_ctx_destroy(ctx);
        ctx = hmcuda_ctx_create(sid, vid);
        if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;
        res = cuDevicePrimaryCtxRetain(&pctx, (CUdevice)device);
        if (res != CUDA_SUCCESS) {
            LOG_ERROR("hmcuda_core_cu_device_primary_ctx_retain: retry failed: %d", res);
            return res;
        }
        LOG("hmcuda_core_cu_device_primary_ctx_retain: retry succeeded after reset");
    }
    /* Store device ordinal in size field for later lookup */
    *ctx_handle = hmcuda_res_add(ctx, RES_TYPE_CONTEXT, (void *)pctx, (uint64_t)device);
    LOG("hmcuda_core_cu_device_primary_ctx_retain: device=%d, ctx=%p, handle=0x%lx",
        device, (void *)pctx, *ctx_handle);

    /* Register guest RAM with CUDA once so D2H DMAs to iov_base are truly async. */
    hmcuda_register_guest_memory();

    return res;
}

uint32_t hmcuda_core_cu_device_primary_ctx_release(uint32_t sid, uint32_t vid,
                                                    int32_t device)
{
    CUresult res = cuDevicePrimaryCtxRelease((CUdevice)device);
    LOG("hmcuda_core_cu_device_primary_ctx_release: device=%d, res=%d", device, res);
    return res;
}

uint32_t hmcuda_core_cu_ctx_set_current(uint32_t sid, uint32_t vid,
                                         uint64_t ctx_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUcontext real_ctx = (CUcontext)hmcuda_res_get(ctx, RES_TYPE_CONTEXT, ctx_handle);
    LOG("hmcuda_core_cu_ctx_set_current: handle=0x%lx, real=%p", ctx_handle, (void *)real_ctx);
    if (!real_ctx)
        LOG_ERROR("hmcuda_core_cu_ctx_set_current: context handle 0x%lx not found — will call cuCtxSetCurrent(NULL)", ctx_handle);
    return cuCtxSetCurrent(real_ctx);
}

uint32_t hmcuda_core_cu_ctx_enable_peer_access(uint32_t sid, uint32_t vid,
                                                uint64_t peer_ctx_handle, uint32_t flags)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUcontext peer = (CUcontext)hmcuda_res_get(ctx, RES_TYPE_CONTEXT, peer_ctx_handle);
    LOG("hmcuda_core_cu_ctx_enable_peer_access: handle=0x%lx, real=%p", peer_ctx_handle, (void *)peer);
    return cuCtxEnablePeerAccess(peer, flags);
}

/* ================================================================== */
/*  Stream management                                                 */
/* ================================================================== */

uint32_t hmcuda_core_cu_stream_create(uint32_t sid, uint32_t vid,
                                       uint32_t flags, uint64_t *stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUstream stream;
    CUresult res = cuStreamCreate(&stream, flags);
    if (res == CUDA_SUCCESS) {
        *stream_handle = hmcuda_res_add(ctx, RES_TYPE_STREAM, (void *)stream, 0);
        LOG("hmcuda_core_cu_stream_create: flags=%u, stream=%p, handle=0x%lx",
            flags, (void *)stream, *stream_handle);
    }
    return res;
}

uint32_t hmcuda_core_cu_stream_destroy(uint32_t sid, uint32_t vid,
                                        uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUstream stream = (CUstream)hmcuda_res_get(ctx, RES_TYPE_STREAM, stream_handle);
    LOG("hmcuda_core_cu_stream_destroy: handle=0x%lx, stream=%p", stream_handle, (void *)stream);
    if (!stream) return CUDA_ERROR_INVALID_HANDLE;

    CUresult res = cuStreamDestroy(stream);
    hmcuda_res_remove(ctx, RES_TYPE_STREAM, stream_handle);
    return res;
}

uint32_t hmcuda_core_cu_stream_synchronize(uint32_t sid, uint32_t vid,
                                            uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) {
        LOG_ERROR("hmcuda_core_cu_stream_synchronize: ctx NULL for sid=%u vid=%u", sid, vid);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUstream stream = resolve_stream(ctx, stream_handle);
    LOG("hmcuda_core_cu_stream_synchronize: handle=0x%lx real_stream=%p",
        stream_handle, (void *)stream);
    CUresult res = cuStreamSynchronize(stream);
    if (res != CUDA_SUCCESS)
        LOG_ERROR("hmcuda_core_cu_stream_synchronize: cuStreamSynchronize returned %d", res);
    return res;
}

uint32_t hmcuda_core_cu_stream_query(uint32_t sid, uint32_t vid,
                                      uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUstream stream = (CUstream)hmcuda_res_get(ctx, RES_TYPE_STREAM, stream_handle);
    return cuStreamQuery(stream);
}

uint32_t hmcuda_core_cu_stream_wait_event(uint32_t sid, uint32_t vid,
                                           uint64_t stream_handle, uint64_t event_handle,
                                           uint32_t flags)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUstream stream = (CUstream)hmcuda_res_get(ctx, RES_TYPE_STREAM, stream_handle);
    CUevent event = (CUevent)hmcuda_res_get(ctx, RES_TYPE_EVENT, event_handle);
    return cuStreamWaitEvent(stream, event, flags);
}

uint32_t hmcuda_core_cu_stream_get_ctx(uint32_t sid, uint32_t vid,
                                        uint64_t stream_handle, uint64_t *ctx_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUstream stream = (CUstream)hmcuda_res_get(ctx, RES_TYPE_STREAM, stream_handle);
    CUcontext sctx;
    CUresult res = cuStreamGetCtx(stream, &sctx);
    if (res == CUDA_SUCCESS) {
        *ctx_handle = hmcuda_res_get_handle(ctx, RES_TYPE_CONTEXT, (void *)sctx);
        if (*ctx_handle == 0) {
            /* Context not yet tracked — add it */
            *ctx_handle = hmcuda_res_add(ctx, RES_TYPE_CONTEXT, (void *)sctx, 0);
        }
    }
    return res;
}

/* ================================================================== */
/*  Event management                                                  */
/* ================================================================== */

uint32_t hmcuda_core_cu_event_create(uint32_t sid, uint32_t vid,
                                      uint32_t flags, uint64_t *event_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUevent event;
    CUresult res = cuEventCreate(&event, flags);
    if (res == CUDA_SUCCESS) {
        *event_handle = hmcuda_res_add(ctx, RES_TYPE_EVENT, (void *)event, 0);
        LOG("hmcuda_core_cu_event_create: flags=%u, event=%p, handle=0x%lx",
            flags, (void *)event, *event_handle);
    }
    return res;
}

uint32_t hmcuda_core_cu_event_destroy(uint32_t sid, uint32_t vid,
                                       uint64_t event_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUevent event = (CUevent)hmcuda_res_get(ctx, RES_TYPE_EVENT, event_handle);
    if (!event) return CUDA_ERROR_INVALID_HANDLE;

    CUresult res = cuEventDestroy(event);
    hmcuda_res_remove(ctx, RES_TYPE_EVENT, event_handle);
    return res;
}

uint32_t hmcuda_core_cu_event_record(uint32_t sid, uint32_t vid,
                                      uint64_t event_handle, uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUevent event = (CUevent)hmcuda_res_get(ctx, RES_TYPE_EVENT, event_handle);
    CUstream stream = (CUstream)hmcuda_res_get(ctx, RES_TYPE_STREAM, stream_handle);
    return cuEventRecord(event, stream);
}

uint32_t hmcuda_core_cu_event_elapsed_time(uint32_t sid, uint32_t vid,
                                            uint64_t start_handle, uint64_t end_handle,
                                            float *ms)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUevent start = (CUevent)hmcuda_res_get(ctx, RES_TYPE_EVENT, start_handle);
    CUevent end = (CUevent)hmcuda_res_get(ctx, RES_TYPE_EVENT, end_handle);
    return cuEventElapsedTime(ms, start, end);
}

/* ================================================================== */
/*  Memory allocation                                                 */
/* ================================================================== */

uint32_t hmcuda_core_cu_mem_alloc(uint32_t sid, uint32_t vid,
                                   uint64_t bytesize, uint64_t *handle_out)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUdeviceptr dptr;
    CUresult res = cuMemAlloc(&dptr, (size_t)bytesize);
    if (res == CUDA_SUCCESS) {
        *handle_out = (uint64_t)dptr;
        hmcuda_res_add_with_handle(ctx, RES_TYPE_MEM, (void *)dptr, bytesize, *handle_out);
        LOG("hmcuda_core_cu_mem_alloc: size=%lu, dptr=%p, handle=0x%lx",
            bytesize, (void *)dptr, *handle_out);
    }
    return res;
}

uint32_t hmcuda_core_cu_mem_free(uint32_t sid, uint32_t vid, uint64_t handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *real_ptr = hmcuda_res_get(ctx, RES_TYPE_MEM, handle);
    if (!real_ptr) return CUDA_ERROR_INVALID_VALUE;

    LOG("hmcuda_core_cu_mem_free: handle=0x%lx, real_ptr=%p", handle, real_ptr);
    CUresult res = cuMemFree((CUdeviceptr)real_ptr);
    hmcuda_res_remove(ctx, RES_TYPE_MEM, handle);
    return res;
}

uint32_t hmcuda_core_cu_mem_host_alloc_vq(uint32_t sid, uint32_t vid,
                                           uint64_t bytesize, uint32_t flags,
                                           uint64_t guest_va, uint64_t byte_offset,
                                           void *host_ptr, size_t batch_size)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    /* On the first batch, allocate the real pinned buffer and register it. */
    if (byte_offset == 0) {
        void *real_ptr;
        /* Always add DEVICEMAP so the GPU can DMA directly to real_ptr (D2H staging). */
        CUresult res = cuMemHostAlloc(&real_ptr, (size_t)bytesize,
                                      flags | CU_MEMHOSTALLOC_DEVICEMAP);
        if (res != CUDA_SUCCESS) return res;
        hmcuda_res_add_with_handle(ctx, RES_TYPE_HOST_MEM, real_ptr, bytesize, guest_va);
        LOG("hmcuda_core_cu_mem_host_alloc_vq: size=%lu, real_ptr=%p, guest_va=0x%lx",
            bytesize, real_ptr, guest_va);

        CUdeviceptr dev_ptr = 0;
        res = cuMemHostGetDevicePointer(&dev_ptr, real_ptr, 0);
        if (res == CUDA_SUCCESS) {
            hmcuda_res_add_with_handle(ctx, RES_TYPE_HOST_DEV_PTR,
                                       (void *)(uintptr_t)dev_ptr, bytesize, guest_va);
            LOG("hmcuda_core_cu_mem_host_alloc_vq: dev_ptr=0x%llx for guest_va=0x%lx",
                (unsigned long long)dev_ptr, guest_va);
        } else {
            LOG("hmcuda_core_cu_mem_host_alloc_vq: cuMemHostGetDevicePointer failed: %d", res);
        }
    }

    /* Copy this batch's data into real_ptr at the right offset. */
    void *real_ptr = hmcuda_res_get(ctx, RES_TYPE_HOST_MEM, guest_va);
    if (!real_ptr) return CUDA_ERROR_INVALID_VALUE;
    memcpy((char *)real_ptr + byte_offset, host_ptr, batch_size);
    return CUDA_SUCCESS;
}

uint32_t hmcuda_core_cu_mem_free_host(uint32_t sid, uint32_t vid, uint64_t guest_va)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *real_ptr = hmcuda_res_get(ctx, RES_TYPE_HOST_MEM, guest_va);
    if (!real_ptr) return CUDA_ERROR_INVALID_VALUE;

    LOG("hmcuda_core_cu_mem_free_host: guest_va=0x%lx, real_ptr=%p", guest_va, real_ptr);
    CUresult res = cuMemFreeHost(real_ptr);
    hmcuda_res_remove(ctx, RES_TYPE_HOST_MEM, guest_va);
    hmcuda_res_remove(ctx, RES_TYPE_HOST_DEV_PTR, guest_va);
    return res;
}

uint32_t hmcuda_core_cu_memset_d32(uint32_t sid, uint32_t vid,
                                    uint64_t dptr_handle, uint32_t value, uint64_t count)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *real_ptr = hmcuda_res_get(ctx, RES_TYPE_MEM, dptr_handle);
    if (!real_ptr) return CUDA_ERROR_INVALID_VALUE;

    return cuMemsetD32((CUdeviceptr)real_ptr, value, (size_t)count);
}

/* ================================================================== */
/*  Memory copy                                                       */
/* ================================================================== */

uint32_t hmcuda_core_cu_memcpy_async(uint32_t sid, uint32_t vid,
                                      uint64_t dst_handle, uint64_t src_handle,
                                      uint64_t bytesize, uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *dst = resolve_any_mem(ctx, dst_handle);
    void *src = resolve_any_mem(ctx, src_handle);
    CUstream stream = (CUstream)hmcuda_res_get(ctx, RES_TYPE_STREAM, stream_handle);

    if (!dst || !src) {
        LOG_ERROR("hmcuda_core_cu_memcpy_async: failed to resolve dst=0x%lx(%p) src=0x%lx(%p)",
                  dst_handle, dst, src_handle, src);
        return CUDA_ERROR_INVALID_VALUE;
    }

    return cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src,
                         (size_t)bytesize, stream);
}

uint32_t hmcuda_core_cu_memcpy_htod_vq(uint32_t sid, uint32_t vid,
                                        uint64_t dst_handle, void *host_ptr,
                                        uint64_t bytesize, uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    if (!host_ptr) return CUDA_ERROR_INVALID_VALUE;

    CUstream stream = resolve_stream(ctx, stream_handle);

    /* Try device memory first; fall back to pinned host memory (e.g. cuMemHostAlloc
     * buffer used as dst so we can initialise real_ptr from CPU-written guest_va data). */
    void *dst = hmcuda_res_get(ctx, RES_TYPE_MEM, dst_handle);
    int dst_is_host_mem = 0;
    if (!dst) {
        dst = hmcuda_res_get(ctx, RES_TYPE_HOST_MEM, dst_handle);
        dst_is_host_mem = (dst != NULL);
    }
    if (!dst) {
        LOG_ERROR("hmcuda_core_cu_memcpy_htod_vq: failed to resolve dst=0x%lx host=%p",
                  dst_handle, host_ptr);
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (dst_is_host_mem &&
        hmcuda_buffer_needs_host_ptr_patch(ctx, host_ptr, (size_t)bytesize)) {
        void *patch_buf = malloc((size_t)bytesize);
        if (!patch_buf)
            return CUDA_ERROR_OUT_OF_MEMORY;

        memcpy(patch_buf, host_ptr, (size_t)bytesize);
        hmcuda_patch_embedded_host_ptrs(ctx, patch_buf, (size_t)bytesize);
        CUresult res;
        if (dst_is_host_mem) {
            memcpy(dst, patch_buf, (size_t)bytesize);
            res = CUDA_SUCCESS;
        } else {
            res = cuMemcpyHtoDAsync((CUdeviceptr)dst, patch_buf, (size_t)bytesize, stream);
            if (res == CUDA_SUCCESS) res = cuStreamSynchronize(stream);
        }
        free(patch_buf);
        return res;
    }

    if (dst_is_host_mem) {
        memcpy(dst, host_ptr, (size_t)bytesize);
        return CUDA_SUCCESS;
    }
    return cuMemcpyHtoDAsync((CUdeviceptr)dst, host_ptr, (size_t)bytesize, stream);
}

/*
 * Return the CUDA-registered staging pointer for a guest destination VA, or
 * NULL if the destination was not allocated via cuMemHostAlloc.  The returned
 * pointer already includes the byte offset within the allocation (range lookup).
 */
void *hmcuda_core_get_host_mem_staging(uint32_t sid, uint32_t vid, uint64_t guest_dst_va)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return NULL;
    return hmcuda_res_get(ctx, RES_TYPE_HOST_MEM, guest_dst_va);
}

/*
 * Submit a single async DMA from a device allocation into a CUDA-registered
 * (pinned) host staging buffer.  The caller is responsible for the subsequent
 * stream synchronize and for scattering staging_ptr into the guest iov pages.
 */
uint32_t hmcuda_core_cu_memcpy_dtoh_staged(uint32_t sid, uint32_t vid,
                                            uint64_t src_handle,
                                            void *staging_ptr,
                                            uint64_t bytesize,
                                            uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUstream stream = resolve_stream(ctx, stream_handle);
    void *src = hmcuda_res_get(ctx, RES_TYPE_MEM, src_handle);
    if (!src || !staging_ptr) {
        LOG_ERROR("hmcuda_core_cu_memcpy_dtoh_staged: src=0x%lx(%p) staging=%p",
                  src_handle, src, staging_ptr);
        return CUDA_ERROR_INVALID_VALUE;
    }

    return cuMemcpyDtoHAsync(staging_ptr, (CUdeviceptr)src, (size_t)bytesize, stream);
}

/* Option A variant: blocking cuMemcpyDtoH — no stream, no separate sync call needed. */
uint32_t hmcuda_core_cu_memcpy_dtoh_staged_sync(uint32_t sid, uint32_t vid,
                                                 uint64_t src_handle,
                                                 void *staging_ptr,
                                                 uint64_t bytesize)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *src = hmcuda_res_get(ctx, RES_TYPE_MEM, src_handle);
    if (!src || !staging_ptr) {
        LOG_ERROR("hmcuda_core_cu_memcpy_dtoh_staged_sync: src=0x%lx(%p) staging=%p",
                  src_handle, src, staging_ptr);
        return CUDA_ERROR_INVALID_VALUE;
    }

    return cuMemcpyDtoH(staging_ptr, (CUdeviceptr)src, (size_t)bytesize);
}

uint32_t hmcuda_core_cu_memcpy_dtoh_vq(uint32_t sid, uint32_t vid,
                                        uint64_t src_handle, void *host_ptr,
                                        uint64_t bytesize, uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUstream stream = resolve_stream(ctx, stream_handle);

    void *src = hmcuda_res_get(ctx, RES_TYPE_MEM, src_handle);
    int src_is_host_mem = 0;
    if (!src) {
        src = hmcuda_res_get(ctx, RES_TYPE_HOST_MEM, src_handle);
        src_is_host_mem = (src != NULL);
    }
    if (!src || !host_ptr) {
        LOG_ERROR("hmcuda_core_cu_memcpy_dtoh_vq: failed to resolve src=0x%lx(%p) host=%p",
                  src_handle, src, host_ptr);
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (src_is_host_mem) {
        LOG("hmcuda_core_cu_memcpy_dtoh_vq: HOST_MEM src=0x%lx real_ptr=%p -> guest=%p bytes=%lu",
            src_handle, src, host_ptr, bytesize);
        memcpy(host_ptr, src, (size_t)bytesize);
        return CUDA_SUCCESS;
    }

    return cuMemcpyDtoHAsync(host_ptr, (CUdeviceptr)src, (size_t)bytesize, stream);
}

uint32_t hmcuda_core_cu_memcpy_htod(uint32_t sid, uint32_t vid,
                                     uint64_t dst_handle, uint64_t src_handle,
                                     uint64_t bytesize)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *dst = hmcuda_res_get(ctx, RES_TYPE_MEM, dst_handle);
    void *src = hmcuda_res_get(ctx, RES_TYPE_HOST_MEM, src_handle);

    if (!dst || !src) {
        LOG_ERROR("hmcuda_core_cu_memcpy_htod: failed to resolve dst=0x%lx(%p) src=0x%lx(%p)",
                  dst_handle, dst, src_handle, src);
        return CUDA_ERROR_INVALID_VALUE;
    }

    return cuMemcpyHtoD((CUdeviceptr)dst, src, (size_t)bytesize);
}

/* ================================================================== */
/*  Pointer query                                                     */
/* ================================================================== */

uint32_t hmcuda_core_cu_pointer_get_attribute(uint32_t sid, uint32_t vid,
                                               uint32_t attrib, uint64_t ptr_handle,
                                               uint64_t *value)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *real_ptr = resolve_any_mem(ctx, ptr_handle);
    if (!real_ptr) return CUDA_ERROR_INVALID_VALUE;

    *value = 0;
    return cuPointerGetAttribute(value, (CUpointer_attribute)attrib,
                                 (CUdeviceptr)real_ptr);
}

/* ================================================================== */
/*  Virtual memory management                                         */
/* ================================================================== */

uint32_t hmcuda_core_cu_mem_create(uint32_t sid, uint32_t vid,
                                    uint64_t size, const CUmemAllocationProp *prop,
                                    uint64_t flags, uint64_t *handle_out)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUmemGenericAllocationHandle alloc_handle;
    CUresult res = cuMemCreate(&alloc_handle, (size_t)size, prop, flags);
    if (res == CUDA_SUCCESS) {
        *handle_out = hmcuda_res_add(ctx, RES_TYPE_GENERIC_ALLOC,
                                     (void *)(uintptr_t)alloc_handle, size);
        LOG("hmcuda_core_cu_mem_create: size=%lu, alloc=%llu, handle=0x%lx",
            size, (unsigned long long)alloc_handle, *handle_out);
    }
    return res;
}

uint32_t hmcuda_core_cu_mem_release(uint32_t sid, uint32_t vid, uint64_t handle)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *raw = hmcuda_res_get(ctx, RES_TYPE_GENERIC_ALLOC, handle);
    if (!raw) return CUDA_ERROR_INVALID_VALUE;

    CUresult res = cuMemRelease((CUmemGenericAllocationHandle)(uintptr_t)raw);
    hmcuda_res_remove(ctx, RES_TYPE_GENERIC_ALLOC, handle);
    return res;
}

uint32_t hmcuda_core_cu_mem_map(uint32_t sid, uint32_t vid,
                                 uint64_t va_handle, uint64_t size, uint64_t offset,
                                 uint64_t alloc_handle, uint64_t flags)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *va = hmcuda_res_get(ctx, RES_TYPE_MEM, va_handle);
    void *alloc = hmcuda_res_get(ctx, RES_TYPE_GENERIC_ALLOC, alloc_handle);
    if (!va) return CUDA_ERROR_INVALID_VALUE;

    return cuMemMap((CUdeviceptr)va, (size_t)size, (size_t)offset,
                    (CUmemGenericAllocationHandle)(uintptr_t)alloc, flags);
}

uint32_t hmcuda_core_cu_mem_unmap(uint32_t sid, uint32_t vid,
                                   uint64_t va_handle, uint64_t size)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *va = hmcuda_res_get(ctx, RES_TYPE_MEM, va_handle);
    if (!va) return CUDA_ERROR_INVALID_VALUE;

    return cuMemUnmap((CUdeviceptr)va, (size_t)size);
}

uint32_t hmcuda_core_cu_mem_address_reserve(uint32_t sid, uint32_t vid,
                                             uint64_t size, uint64_t alignment,
                                             uint64_t addr_hint, uint64_t flags,
                                             uint64_t *handle_out)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUdeviceptr ptr;
    CUresult res = cuMemAddressReserve(&ptr, (size_t)size, (size_t)alignment,
                                       (CUdeviceptr)addr_hint, flags);
    if (res == CUDA_SUCCESS) {
        *handle_out = (uint64_t)ptr;
        hmcuda_res_add_with_handle(ctx, RES_TYPE_MEM, (void *)ptr, size, *handle_out);
        LOG("hmcuda_core_cu_mem_address_reserve: size=%lu, va=%p, handle=0x%lx",
            size, (void *)ptr, *handle_out);
    }
    return res;
}

uint32_t hmcuda_core_cu_mem_address_free(uint32_t sid, uint32_t vid,
                                          uint64_t va_handle, uint64_t size)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *va = hmcuda_res_get(ctx, RES_TYPE_MEM, va_handle);
    if (!va) return CUDA_ERROR_INVALID_VALUE;

    CUresult res = cuMemAddressFree((CUdeviceptr)va, (size_t)size);
    hmcuda_res_remove(ctx, RES_TYPE_MEM, va_handle);
    return res;
}

uint32_t hmcuda_core_cu_mem_set_access(uint32_t sid, uint32_t vid,
                                        uint64_t va_handle, uint64_t size,
                                        const CUmemAccessDesc *descs, size_t count)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *va = hmcuda_res_get(ctx, RES_TYPE_MEM, va_handle);
    if (!va) return CUDA_ERROR_INVALID_VALUE;

    return cuMemSetAccess((CUdeviceptr)va, (size_t)size, descs, count);
}

uint32_t hmcuda_core_cu_mem_export(uint32_t sid, uint32_t vid,
                                    uint64_t alloc_handle, uint32_t handle_type,
                                    uint64_t flags, void *shareable_handle_out)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *raw = hmcuda_res_get(ctx, RES_TYPE_GENERIC_ALLOC, alloc_handle);
    if (!raw) return CUDA_ERROR_INVALID_VALUE;

    return cuMemExportToShareableHandle(
        shareable_handle_out,
        (CUmemGenericAllocationHandle)(uintptr_t)raw,
        (CUmemAllocationHandleType)handle_type, flags);
}

uint32_t hmcuda_core_cu_mem_import(uint32_t sid, uint32_t vid,
                                    const void *shareable_handle,
                                    uint32_t handle_type, uint64_t *handle_out)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUmemGenericAllocationHandle alloc_handle;
    CUresult res = cuMemImportFromShareableHandle(
        &alloc_handle, (void *)shareable_handle,
        (CUmemAllocationHandleType)handle_type);
    if (res == CUDA_SUCCESS) {
        *handle_out = hmcuda_res_add(ctx, RES_TYPE_GENERIC_ALLOC,
                                     (void *)(uintptr_t)alloc_handle, 0);
    }
    return res;
}

/* ================================================================== */
/*  Multicast memory                                                  */
/* ================================================================== */

uint32_t hmcuda_core_cu_multicast_create(uint32_t sid, uint32_t vid,
                                          const CUmulticastObjectProp *prop,
                                          uint64_t *handle_out)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUmemGenericAllocationHandle mc_handle;
    CUresult res = cuMulticastCreate(&mc_handle, prop);
    if (res == CUDA_SUCCESS) {
        *handle_out = hmcuda_res_add(ctx, RES_TYPE_GENERIC_ALLOC,
                                     (void *)(uintptr_t)mc_handle, 0);
    }
    return res;
}

uint32_t hmcuda_core_cu_multicast_add_device(uint32_t sid, uint32_t vid,
                                              uint64_t mc_handle_guest, int32_t device)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *raw = hmcuda_res_get(ctx, RES_TYPE_GENERIC_ALLOC, mc_handle_guest);
    if (!raw) return CUDA_ERROR_INVALID_VALUE;

    return cuMulticastAddDevice((CUmemGenericAllocationHandle)(uintptr_t)raw,
                                (CUdevice)device);
}

uint32_t hmcuda_core_cu_multicast_bind_mem(uint32_t sid, uint32_t vid,
                                            uint64_t mc_handle_guest,
                                            uint64_t mc_offset,
                                            uint64_t mem_handle_guest,
                                            uint64_t mem_offset,
                                            uint64_t size, uint64_t flags)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *mc_raw = hmcuda_res_get(ctx, RES_TYPE_GENERIC_ALLOC, mc_handle_guest);
    void *mem_raw = hmcuda_res_get(ctx, RES_TYPE_GENERIC_ALLOC, mem_handle_guest);
    if (!mc_raw || !mem_raw) return CUDA_ERROR_INVALID_VALUE;

    return cuMulticastBindMem(
        (CUmemGenericAllocationHandle)(uintptr_t)mc_raw,
        (size_t)mc_offset,
        (CUmemGenericAllocationHandle)(uintptr_t)mem_raw,
        (size_t)mem_offset, (size_t)size, flags);
}

uint32_t hmcuda_core_cu_multicast_unbind(uint32_t sid, uint32_t vid,
                                          uint64_t mc_handle_guest, int32_t device,
                                          uint64_t mc_offset, uint64_t size)
{
    Context *ctx = hmcuda_ctx_get(sid, vid);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    void *mc_raw = hmcuda_res_get(ctx, RES_TYPE_GENERIC_ALLOC, mc_handle_guest);
    if (!mc_raw) return CUDA_ERROR_INVALID_VALUE;

    return cuMulticastUnbind(
        (CUmemGenericAllocationHandle)(uintptr_t)mc_raw,
        (CUdevice)device, (size_t)mc_offset, (size_t)size);
}
