#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <cuda.h>

#include "hmcuda_api.h"
#include "hmcuda_log.h"
#include "hmcuda_transport.h"
#include "hmcuda_dispatch.h"

/*
 * CUDA Driver API interception library.
 *
 * Built as libcuda.so.1 — applications that link -lcuda will load this
 * instead of the real driver, forwarding calls through hmcuda transport.
 */

static CUresult hmcuda_drv_dispatch(uint32_t cmd, const void *req, size_t req_len,
                                     void *resp, size_t resp_len)
{
    if (!hmcuda_transport_ready())
        hmcuda_transport_init();
    return (CUresult)hmcuda_transport_dispatch(cmd, req, req_len, resp, resp_len);
}

/* Short aliases — bake in the dispatch function for this file */
#define DRV_FWD(cmd, req)       HMCUDA_FORWARD(hmcuda_drv_dispatch, cmd, req)
#define DRV_FWD_NOREQ(cmd)     HMCUDA_FORWARD_NOREQ(hmcuda_drv_dispatch, cmd)

#define DRV_FWD_RESP(cmd, req, resp_t, out, cast, field) \
    HMCUDA_FORWARD_RESP(hmcuda_drv_dispatch, cmd, req, resp_t, out, cast, field)

#define DRV_FWD_NOREQ_RESP(cmd, resp_t, out, cast, field) \
    HMCUDA_FORWARD_NOREQ_RESP(hmcuda_drv_dispatch, cmd, resp_t, out, cast, field)

/*
 * Set of all live device pointer handles (returned by cuMemAlloc,
 * cuMemAddressReserve, etc.).  Used by cuMemcpyAsync to determine transfer
 * direction: a pointer present here is a backend handle (device), any other
 * value is treated as a guest host address.
 */
static std::unordered_set<uint64_t> g_device_handles;
static bool g_driver_initialized = false;

static CUresult hmcuda_driver_ensure_initialized()
{
    if (g_driver_initialized)
        return CUDA_SUCCESS;

    if (!hmcuda_transport_ready())
        hmcuda_transport_init();

    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_INIT, nullptr, 0, nullptr, 0);
    if (err != CUDA_SUCCESS)
        return err;

    hmcuda_cu_init_req req = { .flags = 0 };
    err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_INIT, &req, sizeof(req), nullptr, 0);
    if (err == CUDA_SUCCESS)
        g_driver_initialized = true;
    return err;
}

static void __attribute__((constructor(102))) hmcuda_driver_init() {
    hmcuda_transport_init();
    hmcuda_driver_ensure_initialized();
}

static void __attribute__((destructor)) hmcuda_driver_fini() {
    hmcuda_transport_fini();
}

extern "C" {

/* ------------------------------------------------------------------ */
/*  Initialization & version                                          */
/* ------------------------------------------------------------------ */

CUresult cuInit(unsigned int flags) {
    HMCUDA_LOG_INFO("cuInit(flags=%u)", flags);
    CUresult err = hmcuda_driver_ensure_initialized();
    if (err != CUDA_SUCCESS)
        return err;
    hmcuda_cu_init_req req = { .flags = flags };
    err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_INIT, &req, sizeof(req), nullptr, 0);
    if (err == CUDA_SUCCESS)
        g_driver_initialized = true;
    return err;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    HMCUDA_LOG_INFO("cuDriverGetVersion");
    DRV_FWD_NOREQ_RESP(HMCUDA_CMD_CU_DRIVER_GET_VERSION,
        hmcuda_cu_driver_get_version_resp, driverVersion, (int), version);
}

/* ------------------------------------------------------------------ */
/*  Device management                                                 */
/* ------------------------------------------------------------------ */

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    HMCUDA_LOG_INFO("cuDeviceGet(ordinal=%d)", ordinal);
    hmcuda_cu_device_get_req req = { .ordinal = ordinal };
    DRV_FWD_RESP(HMCUDA_CMD_CU_DEVICE_GET, req,
        hmcuda_cu_device_get_resp, device, (CUdevice), device);
}

CUresult cuDeviceGetCount(int *count) {
    HMCUDA_LOG_INFO("cuDeviceGetCount");
    DRV_FWD_NOREQ_RESP(HMCUDA_CMD_CU_DEVICE_GET_COUNT,
        hmcuda_cu_device_get_count_resp, count, (int), count);
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    HMCUDA_LOG_INFO("cuDeviceGetName(dev=%d, len=%d)", (int)dev, len);
    hmcuda_cu_device_get_name_req req = {
        .device = (int32_t)dev,
        .name_len = len
    };
    /* Response: fixed header + variable-length name */
    size_t max_resp = sizeof(hmcuda_cu_device_get_name_resp) + len;
    std::vector<uint8_t> resp_buf(max_resp, 0);
    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_DEVICE_GET_NAME,
                                        &req, sizeof(req), resp_buf.data(), max_resp);
    if (err == CUDA_SUCCESS && name) {
        hmcuda_cu_device_get_name_resp *resp = (hmcuda_cu_device_get_name_resp *)resp_buf.data();
        int copy_len = resp->name_len < len ? resp->name_len : len;
        memcpy(name, resp_buf.data() + sizeof(*resp), copy_len);
        if (copy_len < len) name[copy_len] = '\0';
    }
    return err;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    HMCUDA_LOG_DEBUG("cuDeviceGetAttribute(attrib=%d, dev=%d)", (int)attrib, (int)dev);
    hmcuda_cu_device_get_attribute_req req = {
        .attrib = (int32_t)attrib,
        .device = (int32_t)dev
    };
    DRV_FWD_RESP(HMCUDA_CMD_CU_DEVICE_GET_ATTRIBUTE, req,
        hmcuda_cu_device_get_attribute_resp, pi, (int), value);
}

CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    HMCUDA_LOG_INFO("cuDeviceGetUuid(dev=%d)", (int)dev);
    hmcuda_cu_device_get_uuid_req req = { .device = (int32_t)dev };
    hmcuda_cu_device_get_uuid_resp resp;
    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_DEVICE_GET_UUID,
                                        &req, sizeof(req), &resp, sizeof(resp));
    if (err == CUDA_SUCCESS && uuid)
        memcpy(uuid->bytes, resp.uuid, 16);
    return err;
}

CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    HMCUDA_LOG_INFO("cuDeviceCanAccessPeer(dev=%d, peer=%d)", (int)dev, (int)peerDev);
    hmcuda_cu_device_can_access_peer_req req = {
        .device = (int32_t)dev,
        .peer_device = (int32_t)peerDev
    };
    DRV_FWD_RESP(HMCUDA_CMD_CU_DEVICE_CAN_ACCESS_PEER, req,
        hmcuda_cu_device_can_access_peer_resp, canAccessPeer, (int), can_access);
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    HMCUDA_LOG_INFO("cuDevicePrimaryCtxRetain(dev=%d)", (int)dev);
    hmcuda_cu_device_primary_ctx_retain_req req = { .device = (int32_t)dev };
    DRV_FWD_RESP(HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RETAIN, req,
        hmcuda_cu_device_primary_ctx_retain_resp, pctx, (CUcontext)(uintptr_t), context);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    HMCUDA_LOG_INFO("cuDevicePrimaryCtxRelease(dev=%d)", (int)dev);
    hmcuda_cu_device_primary_ctx_release_req req = { .device = (int32_t)dev };
    DRV_FWD(HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RELEASE, req);
}

/* ------------------------------------------------------------------ */
/*  Context management                                                */
/* ------------------------------------------------------------------ */

CUresult cuCtxSetCurrent(CUcontext ctx) {
    HMCUDA_LOG_DEBUG("cuCtxSetCurrent(ctx=%p)", (void*)ctx);
    hmcuda_cu_ctx_set_current_req req = { .context = (uint64_t)(uintptr_t)ctx };
    DRV_FWD(HMCUDA_CMD_CU_CTX_SET_CURRENT, req);
}

CUresult cuCtxGetDevice(CUdevice *device) {
    HMCUDA_LOG_DEBUG("cuCtxGetDevice");
    DRV_FWD_NOREQ_RESP(HMCUDA_CMD_CU_CTX_GET_DEVICE,
        hmcuda_cu_ctx_get_device_resp, device, (CUdevice), device);
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags) {
    HMCUDA_LOG_INFO("cuCtxEnablePeerAccess(peer=%p, flags=%u)", (void*)peerContext, flags);
    hmcuda_cu_ctx_enable_peer_access_req req = {
        .peer_context = (uint64_t)(uintptr_t)peerContext,
        .flags = flags
    };
    DRV_FWD(HMCUDA_CMD_CU_CTX_ENABLE_PEER_ACCESS, req);
}

/* ------------------------------------------------------------------ */
/*  Stream management                                                 */
/* ------------------------------------------------------------------ */

CUresult cuStreamCreate(CUstream *phStream, unsigned int flags) {
    HMCUDA_LOG_INFO("cuStreamCreate(flags=%u)", flags);
    hmcuda_cu_stream_create_req req = { .flags = flags };
    DRV_FWD_RESP(HMCUDA_CMD_CU_STREAM_CREATE, req,
        hmcuda_cu_stream_create_resp, phStream, (CUstream)(uintptr_t), stream);
}

CUresult cuStreamDestroy(CUstream hStream) {
    HMCUDA_LOG_INFO("cuStreamDestroy(stream=%p)", (void*)hStream);
    hmcuda_cu_stream_destroy_req req = { .stream = (uint64_t)(uintptr_t)hStream };
    DRV_FWD(HMCUDA_CMD_CU_STREAM_DESTROY, req);
}

CUresult cuStreamSynchronize(CUstream hStream) {
    HMCUDA_LOG_DEBUG("cuStreamSynchronize(stream=%p)", (void*)hStream);
    hmcuda_cu_stream_synchronize_req req = { .stream = (uint64_t)(uintptr_t)hStream };
    DRV_FWD(HMCUDA_CMD_CU_STREAM_SYNCHRONIZE, req);
}

CUresult cuStreamQuery(CUstream hStream) {
    HMCUDA_LOG_DEBUG("cuStreamQuery(stream=%p)", (void*)hStream);
    hmcuda_cu_stream_query_req req = { .stream = (uint64_t)(uintptr_t)hStream };
    DRV_FWD(HMCUDA_CMD_CU_STREAM_QUERY, req);
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
    HMCUDA_LOG_DEBUG("cuStreamWaitEvent(stream=%p, event=%p)", (void*)hStream, (void*)hEvent);
    hmcuda_cu_stream_wait_event_req req = {
        .stream = (uint64_t)(uintptr_t)hStream,
        .event = (uint64_t)(uintptr_t)hEvent,
        .flags = flags
    };
    DRV_FWD(HMCUDA_CMD_CU_STREAM_WAIT_EVENT, req);
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    HMCUDA_LOG_DEBUG("cuStreamGetCtx(stream=%p)", (void*)hStream);
    hmcuda_cu_stream_get_ctx_req req = { .stream = (uint64_t)(uintptr_t)hStream };
    DRV_FWD_RESP(HMCUDA_CMD_CU_STREAM_GET_CTX, req,
        hmcuda_cu_stream_get_ctx_resp, pctx, (CUcontext)(uintptr_t), context);
}

/* ------------------------------------------------------------------ */
/*  Event management                                                  */
/* ------------------------------------------------------------------ */

CUresult cuEventCreate(CUevent *phEvent, unsigned int flags) {
    HMCUDA_LOG_INFO("cuEventCreate(flags=%u)", flags);
    hmcuda_cu_event_create_req req = { .flags = flags };
    DRV_FWD_RESP(HMCUDA_CMD_CU_EVENT_CREATE, req,
        hmcuda_cu_event_create_resp, phEvent, (CUevent)(uintptr_t), event);
}

CUresult cuEventDestroy(CUevent hEvent) {
    HMCUDA_LOG_INFO("cuEventDestroy(event=%p)", (void*)hEvent);
    hmcuda_cu_event_destroy_req req = { .event = (uint64_t)(uintptr_t)hEvent };
    DRV_FWD(HMCUDA_CMD_CU_EVENT_DESTROY, req);
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    HMCUDA_LOG_DEBUG("cuEventRecord(event=%p, stream=%p)", (void*)hEvent, (void*)hStream);
    hmcuda_cu_event_record_req req = {
        .event = (uint64_t)(uintptr_t)hEvent,
        .stream = (uint64_t)(uintptr_t)hStream
    };
    DRV_FWD(HMCUDA_CMD_CU_EVENT_RECORD, req);
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    HMCUDA_LOG_DEBUG("cuEventElapsedTime(start=%p, end=%p)", (void*)hStart, (void*)hEnd);
    hmcuda_cu_event_elapsed_time_req req = {
        .start_event = (uint64_t)(uintptr_t)hStart,
        .end_event = (uint64_t)(uintptr_t)hEnd
    };
    DRV_FWD_RESP(HMCUDA_CMD_CU_EVENT_ELAPSED_TIME, req,
        hmcuda_cu_event_elapsed_time_resp, pMilliseconds, (float), ms);
}

/* ------------------------------------------------------------------ */
/*  Memory allocation                                                 */
/* ------------------------------------------------------------------ */

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    HMCUDA_LOG_INFO("cuMemAlloc(size=%zu)", bytesize);
    hmcuda_cu_mem_alloc_req req = { .bytesize = bytesize };
    hmcuda_cu_mem_alloc_resp _resp;
    CUresult _err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEM_ALLOC, &req, sizeof(req), &_resp, sizeof(_resp));
    if (!_err && dptr) {
        *dptr = (CUdeviceptr)_resp.dptr;
        g_device_handles.insert(_resp.dptr);
    }
    return _err;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    HMCUDA_LOG_INFO("cuMemFree(dptr=0x%lx)", (unsigned long)dptr);
    g_device_handles.erase((uint64_t)dptr);
    hmcuda_cu_mem_free_req req = { .dptr = (uint64_t)dptr };
    return hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEM_FREE, &req, sizeof(req), nullptr, 0);
}

CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags) {
    HMCUDA_LOG_INFO("cuMemHostAlloc(size=%zu, flags=%u)", bytesize, flags);
    if (!pp) return CUDA_ERROR_INVALID_VALUE;

    /* Allocate a real guest VA so the CPU can read/write it directly. */
    void *guest_va = malloc(bytesize);
    if (!guest_va) return CUDA_ERROR_OUT_OF_MEMORY;

    /* Pin the pages and send them to the backend via the virtqueue page-pin
     * path (same mechanism as H2D cudaMemcpy).  The backend calls real
     * cuMemHostAlloc, copies the page data into real_ptr, and stores the
     * guest_va → real_ptr mapping. */
    hmcuda_cu_mem_host_alloc_req req = {
        .bytesize    = bytesize,
        .flags       = flags,
        .guest_va    = (uint64_t)(uintptr_t)guest_va,
        .byte_offset = 0
    };
    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEM_HOST_ALLOC,
                                        &req, sizeof(req), nullptr, 0);
    if (err != CUDA_SUCCESS) {
        free(guest_va);
        return err;
    }

    hmcuda_transport_host_mem_register(guest_va, bytesize);
    *pp = guest_va;
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost(void *p) {
    HMCUDA_LOG_INFO("cuMemFreeHost(ptr=%p)", p);
    size_t bytesize = hmcuda_transport_host_mem_size(p);
    if (bytesize == 0) return CUDA_ERROR_INVALID_VALUE;

    hmcuda_transport_host_mem_unregister(p);

    hmcuda_cu_mem_free_host_req req = { .guest_va = (uint64_t)(uintptr_t)p };
    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEM_FREE_HOST,
                                        &req, sizeof(req), nullptr, 0);
    free(p);
    return err;
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    HMCUDA_LOG_INFO("cuMemsetD32(dptr=0x%lx, val=%u, N=%zu)", (unsigned long)dstDevice, ui, N);
    hmcuda_cu_memset_d32_req req = {
        .dptr = (uint64_t)dstDevice,
        .value = ui,
        .count = N
    };
    DRV_FWD(HMCUDA_CMD_CU_MEMSET_D32, req);
}

/* ------------------------------------------------------------------ */
/*  Memory copy                                                       */
/* ------------------------------------------------------------------ */

static CUresult hmcuda_sync_dirty_host_src_to_guest(void *host_ptr, size_t bytesize,
                                                    CUstream hStream)
{
    hmcuda_memcpy_req req = {
        .dst    = (uint64_t)(uintptr_t)host_ptr,
        .src    = (uint64_t)(uintptr_t)host_ptr,
        .count  = (uint64_t)bytesize,
        .kind   = 2, /* backend cuMemHostAlloc real_ptr -> guest_va */
        .stream = (uint64_t)(uintptr_t)hStream
    };
    CUresult res = (CUresult)hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEMCPY,
                                                  &req, sizeof(req), nullptr, 0);
    if (res == CUDA_SUCCESS)
        hmcuda_transport_clear_gpu_dirty(host_ptr);
    return res;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    HMCUDA_LOG_INFO("cuMemcpyAsync(dst=0x%lx, src=0x%lx, size=%zu)",
                    (unsigned long)dst, (unsigned long)src, ByteCount);
    bool src_is_device = g_device_handles.count((uint64_t)src) > 0;
    bool dst_is_device = g_device_handles.count((uint64_t)dst) > 0;

    if (!src_is_device && dst_is_device) {
        if (hmcuda_transport_is_gpu_dirty((void *)src)) {
            HMCUDA_LOG_INFO("cuMemcpyAsync: H2D src is GPU-dirty, syncing real_ptr -> guest_va before page-pin path");
            CUresult sync_res = hmcuda_sync_dirty_host_src_to_guest((void *)src,
                                                                    ByteCount, hStream);
            if (sync_res != CUDA_SUCCESS)
                return sync_res;
        }
        HMCUDA_LOG_INFO("cuMemcpyAsync: H2D (host src), routing via page-pin path");
        hmcuda_memcpy_req req = {
            .dst    = (uint64_t)dst,
            .src    = (uint64_t)src,
            .count  = ByteCount,
            .kind   = 1,
            .stream = (uint64_t)(uintptr_t)hStream
        };
        return (CUresult)hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEMCPY, &req, sizeof(req), nullptr, 0);
    }
    if (src_is_device && !dst_is_device) {
        /* D2H: dst is any host pointer.
         * Kernel driver pins the pages and receives data via virtqueue. */
        HMCUDA_LOG_INFO("cuMemcpyAsync: D2H (host dst), routing via page-pin path");
        hmcuda_memcpy_req req = {
            .dst    = (uint64_t)dst,
            .src    = (uint64_t)src,
            .count  = ByteCount,
            .kind   = 2,
            .stream = (uint64_t)(uintptr_t)hStream
        };
        CUresult res = (CUresult)hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEMCPY, &req, sizeof(req), nullptr, 0);
        /* guest_va is now authoritative; clear gpu_dirty so the next
         * pre-launch sync will push guest_va → real_ptr. */
        hmcuda_transport_clear_gpu_dirty((void *)dst);
        return res;
    }
    /* D2D: both are device handles, backend calls cuMemcpyAsync directly. */
    HMCUDA_LOG_INFO("cuMemcpyAsync: D2D, routing via Driver API");
    hmcuda_cu_memcpy_async_req req = {
        .dst     = (uint64_t)dst,
        .src     = (uint64_t)src,
        .bytesize = ByteCount,
        .stream  = (uint64_t)(uintptr_t)hStream
    };
    DRV_FWD(HMCUDA_CMD_CU_MEMCPY_ASYNC, req);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    HMCUDA_LOG_INFO("cuMemcpyHtoD(dst=0x%lx, src=%p, size=%zu)",
                    (unsigned long)dstDevice, srcHost, ByteCount);
    if (hmcuda_transport_is_gpu_dirty(srcHost)) {
        HMCUDA_LOG_INFO("cuMemcpyHtoD: src is GPU-dirty, syncing real_ptr -> guest_va before page-pin path");
        CUresult sync_res = hmcuda_sync_dirty_host_src_to_guest((void *)srcHost,
                                                                ByteCount, 0);
        if (sync_res != CUDA_SUCCESS)
            return sync_res;
    }
    HMCUDA_LOG_INFO("cuMemcpyHtoD: routing via page-pin path");
    hmcuda_memcpy_req req = {
        .dst    = (uint64_t)dstDevice,
        .src    = (uint64_t)(uintptr_t)srcHost,
        .count  = ByteCount,
        .kind   = 1, /* H2D */
        .stream = 0  /* synchronous */
    };
    return (CUresult)hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEMCPY,
                                          &req, sizeof(req), nullptr, 0);
}

CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    HMCUDA_LOG_INFO("cuMemAllocHost(size=%zu) -> cuMemHostAlloc(flags=0)", bytesize);
    return cuMemHostAlloc(pp, bytesize, 0);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    HMCUDA_LOG_INFO("cuMemcpyDtoH(dst=%p, src=0x%lx, size=%zu)",
                    dstHost, (unsigned long)srcDevice, ByteCount);
    hmcuda_memcpy_req req = {
        .dst    = (uint64_t)(uintptr_t)dstHost,
        .src    = (uint64_t)srcDevice,
        .count  = ByteCount,
        .kind   = 2, /* D2H */
        .stream = 0  /* synchronous */
    };
    CUresult res = (CUresult)hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEMCPY, &req, sizeof(req), nullptr, 0);
    hmcuda_transport_clear_gpu_dirty(dstHost);
    return res;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    HMCUDA_LOG_INFO("cuMemcpy(dst=0x%lx, src=0x%lx, size=%zu)",
                    (unsigned long)dst, (unsigned long)src, ByteCount);
    return cuMemcpyAsync(dst, src, ByteCount, 0);
}

/* ------------------------------------------------------------------ */
/*  Pointer & error                                                   */
/* ------------------------------------------------------------------ */

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    HMCUDA_LOG_DEBUG("cuPointerGetAttribute(attrib=%d, ptr=0x%lx)", (int)attribute, (unsigned long)ptr);
    hmcuda_cu_pointer_get_attribute_req req = {
        .attrib = (uint32_t)attribute,
        .ptr = (uint64_t)ptr
    };
    hmcuda_cu_pointer_get_attribute_resp resp;
    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_POINTER_GET_ATTRIBUTE,
                                        &req, sizeof(req), &resp, sizeof(resp));
    if (err == CUDA_SUCCESS && data) {
        /* Most attributes fit in an int/unsigned; copy 8 bytes to cover all */
        memcpy(data, &resp.value, sizeof(resp.value));
    }
    return err;
}

CUresult cuGetErrorString(CUresult error, const char **pStr) {
    HMCUDA_LOG_DEBUG("cuGetErrorString(error=%d)", (int)error);
    hmcuda_cu_get_error_string_req req = { .error = (uint32_t)error };

    uint8_t resp_buf[sizeof(hmcuda_cu_get_error_string_resp) + 256];
    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_GET_ERROR_STRING,
                                        &req, sizeof(req), resp_buf, sizeof(resp_buf));
    if (err == CUDA_SUCCESS && pStr) {
        hmcuda_cu_get_error_string_resp *resp = (hmcuda_cu_get_error_string_resp *)resp_buf;
        if (resp->str_len > 0) {
            /* Return pointer to static thread-local buffer */
            static thread_local char tl_buf[256];
            size_t copy = resp->str_len < sizeof(tl_buf) ? resp->str_len : sizeof(tl_buf) - 1;
            memcpy(tl_buf, resp_buf + sizeof(*resp), copy);
            tl_buf[copy] = '\0';
            *pStr = tl_buf;
        } else {
            *pStr = "Unknown error";
        }
    }
    return err;
}

CUresult cuGetErrorName(CUresult error, const char **pStr) {
    HMCUDA_LOG_DEBUG("cuGetErrorName(error=%d)", (int)error);
    hmcuda_cu_get_error_name_req req = { .error = (uint32_t)error };

    uint8_t resp_buf[sizeof(hmcuda_cu_get_error_name_resp) + 256];
    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_GET_ERROR_NAME,
                                        &req, sizeof(req), resp_buf, sizeof(resp_buf));
    if (err == CUDA_SUCCESS && pStr) {
        hmcuda_cu_get_error_name_resp *resp = (hmcuda_cu_get_error_name_resp *)resp_buf;
        if (resp->str_len > 0) {
            static thread_local char tl_buf[256];
            size_t copy = resp->str_len < sizeof(tl_buf) ? resp->str_len : sizeof(tl_buf) - 1;
            memcpy(tl_buf, resp_buf + sizeof(*resp), copy);
            tl_buf[copy] = '\0';
            *pStr = tl_buf;
        } else {
            *pStr = "UNKNOWN";
        }
    }
    return err;
}

/* ------------------------------------------------------------------ */
/*  Virtual memory management                                         */
/* ------------------------------------------------------------------ */

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags) {
    HMCUDA_LOG_INFO("cuMemCreate(size=%zu)", size);
    hmcuda_cu_mem_create_req req = {
        .size = size,
        .prop = {
            .type = (uint32_t)prop->type,
            .location_type = (uint32_t)prop->location.type,
            .location_id = prop->location.id,
            .requested_handle_types = (uint32_t)prop->requestedHandleTypes
        },
        .flags = flags
    };
    DRV_FWD_RESP(HMCUDA_CMD_CU_MEM_CREATE, req,
        hmcuda_cu_mem_create_resp, handle, (CUmemGenericAllocationHandle), handle);
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    HMCUDA_LOG_INFO("cuMemRelease(handle=0x%lx)", (unsigned long)handle);
    hmcuda_cu_mem_release_req req = { .handle = (uint64_t)handle };
    DRV_FWD(HMCUDA_CMD_CU_MEM_RELEASE, req);
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, unsigned long long flags) {
    HMCUDA_LOG_INFO("cuMemMap(ptr=0x%lx, size=%zu)", (unsigned long)ptr, size);
    hmcuda_cu_mem_map_req req = {
        .ptr = (uint64_t)ptr,
        .size = size,
        .offset = offset,
        .handle = (uint64_t)handle,
        .flags = flags
    };
    DRV_FWD(HMCUDA_CMD_CU_MEM_MAP, req);
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    HMCUDA_LOG_INFO("cuMemUnmap(ptr=0x%lx, size=%zu)", (unsigned long)ptr, size);
    hmcuda_cu_mem_unmap_req req = {
        .ptr = (uint64_t)ptr,
        .size = size
    };
    DRV_FWD(HMCUDA_CMD_CU_MEM_UNMAP, req);
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {
    HMCUDA_LOG_INFO("cuMemAddressReserve(size=%zu)", size);
    hmcuda_cu_mem_address_reserve_req req = {
        .size = size,
        .alignment = alignment,
        .addr = (uint64_t)addr,
        .flags = flags
    };
    hmcuda_cu_mem_address_reserve_resp _resp;
    CUresult _err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEM_ADDRESS_RESERVE, &req, sizeof(req), &_resp, sizeof(_resp));
    if (!_err && ptr) {
        *ptr = (CUdeviceptr)_resp.ptr;
        g_device_handles.insert(_resp.ptr);
    }
    return _err;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    HMCUDA_LOG_INFO("cuMemAddressFree(ptr=0x%lx, size=%zu)", (unsigned long)ptr, size);
    g_device_handles.erase((uint64_t)ptr);
    hmcuda_cu_mem_address_free_req req = {
        .ptr = (uint64_t)ptr,
        .size = size
    };
    return hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEM_ADDRESS_FREE, &req, sizeof(req), nullptr, 0);
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count) {
    HMCUDA_LOG_INFO("cuMemSetAccess(ptr=0x%lx, size=%zu, count=%zu)", (unsigned long)ptr, size, count);
    size_t descs_size = count * sizeof(hmcuda_cu_mem_access_desc);
    std::vector<uint8_t> payload(sizeof(hmcuda_cu_mem_set_access_req) + descs_size);

    hmcuda_cu_mem_set_access_req *req = (hmcuda_cu_mem_set_access_req *)payload.data();
    req->ptr = (uint64_t)ptr;
    req->size = size;
    req->count = (uint32_t)count;

    hmcuda_cu_mem_access_desc *out_descs =
        (hmcuda_cu_mem_access_desc *)(payload.data() + sizeof(*req));
    for (size_t i = 0; i < count; i++) {
        out_descs[i].location_type = (uint32_t)desc[i].location.type;
        out_descs[i].location_id = desc[i].location.id;
        out_descs[i].flags = (uint32_t)desc[i].flags;
    }

    return hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEM_SET_ACCESS,
                                payload.data(), payload.size(), nullptr, 0);
}

CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                        const CUmemAllocationProp *prop,
                                        CUmemAllocationGranularity_flags option) {
    HMCUDA_LOG_INFO("cuMemGetAllocationGranularity");
    hmcuda_cu_mem_get_alloc_granularity_req req = {
        .prop = {
            .type = (uint32_t)prop->type,
            .location_type = (uint32_t)prop->location.type,
            .location_id = prop->location.id,
            .requested_handle_types = (uint32_t)prop->requestedHandleTypes
        },
        .option = (uint32_t)option
    };
    DRV_FWD_RESP(HMCUDA_CMD_CU_MEM_GET_ALLOC_GRANULARITY, req,
        hmcuda_cu_mem_get_alloc_granularity_resp, granularity, (size_t), granularity);
}

CUresult cuMemExportToShareableHandle(void *shareableHandle,
                                       CUmemGenericAllocationHandle handle,
                                       CUmemAllocationHandleType handleType,
                                       unsigned long long flags) {
    HMCUDA_LOG_INFO("cuMemExportToShareableHandle");
    hmcuda_cu_mem_export_req req = {
        .handle = (uint64_t)handle,
        .handle_type = (uint32_t)handleType,
        .flags = flags
    };
    hmcuda_cu_mem_export_resp resp;
    CUresult err = hmcuda_drv_dispatch(HMCUDA_CMD_CU_MEM_EXPORT_TO_SHAREABLE_HANDLE,
                                        &req, sizeof(req), &resp, sizeof(resp));
    if (err == CUDA_SUCCESS && shareableHandle)
        memcpy(shareableHandle, resp.shareable_handle, sizeof(resp.shareable_handle));
    return err;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
                                         void *osHandle,
                                         CUmemAllocationHandleType shHandleType) {
    HMCUDA_LOG_INFO("cuMemImportFromShareableHandle");
    hmcuda_cu_mem_import_req req;
    memset(&req, 0, sizeof(req));
    memcpy(req.shareable_handle, osHandle, sizeof(req.shareable_handle));
    req.handle_type = (uint32_t)shHandleType;

    DRV_FWD_RESP(HMCUDA_CMD_CU_MEM_IMPORT_FROM_SHAREABLE_HANDLE, req,
        hmcuda_cu_mem_import_resp, handle, (CUmemGenericAllocationHandle), handle);
}

/* ------------------------------------------------------------------ */
/*  Multicast memory                                                  */
/* ------------------------------------------------------------------ */

CUresult cuMulticastCreate(CUmemGenericAllocationHandle *mcHandle,
                           const CUmulticastObjectProp *prop) {
    HMCUDA_LOG_INFO("cuMulticastCreate(numDevices=%u)", prop->numDevices);
    hmcuda_cu_multicast_create_req req = {
        .prop = {
            .num_devices = prop->numDevices,
            .handle_types = (uint32_t)prop->handleTypes,
            .size = prop->size
        }
    };
    DRV_FWD_RESP(HMCUDA_CMD_CU_MULTICAST_CREATE, req,
        hmcuda_cu_multicast_create_resp, mcHandle, (CUmemGenericAllocationHandle), mc_handle);
}

CUresult cuMulticastGetGranularity(size_t *granularity,
                                    const CUmulticastObjectProp *prop,
                                    CUmulticastGranularity_flags option) {
    HMCUDA_LOG_INFO("cuMulticastGetGranularity");
    hmcuda_cu_multicast_get_granularity_req req = {
        .prop = {
            .num_devices = prop->numDevices,
            .handle_types = (uint32_t)prop->handleTypes,
            .size = prop->size
        },
        .option = (uint32_t)option
    };
    DRV_FWD_RESP(HMCUDA_CMD_CU_MULTICAST_GET_GRANULARITY, req,
        hmcuda_cu_multicast_get_granularity_resp, granularity, (size_t), granularity);
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev) {
    HMCUDA_LOG_INFO("cuMulticastAddDevice(dev=%d)", (int)dev);
    hmcuda_cu_multicast_add_device_req req = {
        .mc_handle = (uint64_t)mcHandle,
        .device = (int32_t)dev
    };
    DRV_FWD(HMCUDA_CMD_CU_MULTICAST_ADD_DEVICE, req);
}

CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset,
                            CUmemGenericAllocationHandle memHandle, size_t memOffset,
                            size_t size, unsigned long long flags) {
    HMCUDA_LOG_INFO("cuMulticastBindMem(size=%zu)", size);
    hmcuda_cu_multicast_bind_mem_req req = {
        .mc_handle = (uint64_t)mcHandle,
        .mc_offset = mcOffset,
        .mem_handle = (uint64_t)memHandle,
        .mem_offset = memOffset,
        .size = size,
        .flags = flags
    };
    DRV_FWD(HMCUDA_CMD_CU_MULTICAST_BIND_MEM, req);
}

CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev,
                           size_t mcOffset, size_t size) {
    HMCUDA_LOG_INFO("cuMulticastUnbind(dev=%d, size=%zu)", (int)dev, size);
    hmcuda_cu_multicast_unbind_req req = {
        .mc_handle = (uint64_t)mcHandle,
        .device = (int32_t)dev,
        .mc_offset = mcOffset,
        .size = size
    };
    DRV_FWD(HMCUDA_CMD_CU_MULTICAST_UNBIND, req);
}

} // extern "C"
