#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/uio.h>
#include <cuda.h>

#include "hmcuda_api.h"
#include "cmd_dispatch.h"
#include "log.h"

/* Core Driver API prototypes (implemented in core_driver.c) */
extern uint32_t hmcuda_core_cu_device_primary_ctx_retain(uint32_t sid, uint32_t vid, int32_t device, uint64_t *ctx_handle);
extern uint32_t hmcuda_core_cu_device_primary_ctx_release(uint32_t sid, uint32_t vid, int32_t device);
extern uint32_t hmcuda_core_cu_ctx_set_current(uint32_t sid, uint32_t vid, uint64_t ctx_handle);
extern uint32_t hmcuda_core_cu_ctx_enable_peer_access(uint32_t sid, uint32_t vid, uint64_t peer_ctx_handle, uint32_t flags);
extern uint32_t hmcuda_core_cu_stream_create(uint32_t sid, uint32_t vid, uint32_t flags, uint64_t *stream_handle);
extern uint32_t hmcuda_core_cu_stream_destroy(uint32_t sid, uint32_t vid, uint64_t stream_handle);
extern uint32_t hmcuda_core_cu_stream_synchronize(uint32_t sid, uint32_t vid, uint64_t stream_handle);
extern uint32_t hmcuda_core_cu_stream_query(uint32_t sid, uint32_t vid, uint64_t stream_handle);
extern uint32_t hmcuda_core_cu_stream_wait_event(uint32_t sid, uint32_t vid, uint64_t stream_handle, uint64_t event_handle, uint32_t flags);
extern uint32_t hmcuda_core_cu_stream_get_ctx(uint32_t sid, uint32_t vid, uint64_t stream_handle, uint64_t *ctx_handle);
extern uint32_t hmcuda_core_cu_event_create(uint32_t sid, uint32_t vid, uint32_t flags, uint64_t *event_handle);
extern uint32_t hmcuda_core_cu_event_destroy(uint32_t sid, uint32_t vid, uint64_t event_handle);
extern uint32_t hmcuda_core_cu_event_record(uint32_t sid, uint32_t vid, uint64_t event_handle, uint64_t stream_handle);
extern uint32_t hmcuda_core_cu_event_elapsed_time(uint32_t sid, uint32_t vid, uint64_t start_handle, uint64_t end_handle, float *ms);
extern uint32_t hmcuda_core_cu_mem_alloc(uint32_t sid, uint32_t vid, uint64_t bytesize, uint64_t *handle_out);
extern uint32_t hmcuda_core_cu_mem_free(uint32_t sid, uint32_t vid, uint64_t handle);
extern uint32_t hmcuda_core_cu_mem_host_alloc_vq(uint32_t sid, uint32_t vid, uint64_t bytesize,
                                                  uint32_t flags, uint64_t guest_va,
                                                  uint64_t byte_offset, void *host_ptr, size_t batch_size);
extern uint32_t hmcuda_core_cu_mem_free_host(uint32_t sid, uint32_t vid, uint64_t guest_va);
extern uint32_t hmcuda_core_cu_memset_d32(uint32_t sid, uint32_t vid, uint64_t dptr_handle, uint32_t value, uint64_t count);
extern uint32_t hmcuda_core_cu_memcpy_async(uint32_t sid, uint32_t vid, uint64_t dst_handle, uint64_t src_handle, uint64_t bytesize, uint64_t stream_handle);
extern uint32_t hmcuda_core_cu_memcpy_htod(uint32_t sid, uint32_t vid, uint64_t dst_handle, uint64_t src_handle, uint64_t bytesize);
extern uint32_t hmcuda_core_cu_memcpy_htod_vq(uint32_t sid, uint32_t vid, uint64_t dst_handle, void *host_ptr, uint64_t bytesize, uint64_t stream_handle);
extern uint32_t hmcuda_core_cu_memcpy_dtoh_vq(uint32_t sid, uint32_t vid, uint64_t src_handle, void *host_ptr, uint64_t bytesize, uint64_t stream_handle);
extern void    *hmcuda_core_get_host_mem_staging(uint32_t sid, uint32_t vid, uint64_t guest_dst_va);
extern uint32_t hmcuda_core_cu_memcpy_dtoh_staged(uint32_t sid, uint32_t vid, uint64_t src_handle, void *staging_ptr, uint64_t bytesize, uint64_t stream_handle);
extern uint32_t hmcuda_core_cu_memcpy_dtoh_staged_sync(uint32_t sid, uint32_t vid, uint64_t src_handle, void *staging_ptr, uint64_t bytesize);
extern uint32_t hmcuda_core_cu_pointer_get_attribute(uint32_t sid, uint32_t vid, uint32_t attrib, uint64_t ptr_handle, uint64_t *value);
extern uint32_t hmcuda_core_cu_mem_create(uint32_t sid, uint32_t vid, uint64_t size, const CUmemAllocationProp *prop, uint64_t flags, uint64_t *handle_out);
extern uint32_t hmcuda_core_cu_mem_release(uint32_t sid, uint32_t vid, uint64_t handle);
extern uint32_t hmcuda_core_cu_mem_map(uint32_t sid, uint32_t vid, uint64_t va_handle, uint64_t size, uint64_t offset, uint64_t alloc_handle, uint64_t flags);
extern uint32_t hmcuda_core_cu_mem_unmap(uint32_t sid, uint32_t vid, uint64_t va_handle, uint64_t size);
extern uint32_t hmcuda_core_cu_mem_address_reserve(uint32_t sid, uint32_t vid, uint64_t size, uint64_t alignment, uint64_t addr_hint, uint64_t flags, uint64_t *handle_out);
extern uint32_t hmcuda_core_cu_mem_address_free(uint32_t sid, uint32_t vid, uint64_t va_handle, uint64_t size);
extern uint32_t hmcuda_core_cu_mem_set_access(uint32_t sid, uint32_t vid, uint64_t va_handle, uint64_t size, const CUmemAccessDesc *descs, size_t count);
extern uint32_t hmcuda_core_cu_mem_export(uint32_t sid, uint32_t vid, uint64_t alloc_handle, uint32_t handle_type, uint64_t flags, void *shareable_handle_out);
extern uint32_t hmcuda_core_cu_mem_import(uint32_t sid, uint32_t vid, const void *shareable_handle, uint32_t handle_type, uint64_t *handle_out);
extern uint32_t hmcuda_core_cu_multicast_create(uint32_t sid, uint32_t vid, const CUmulticastObjectProp *prop, uint64_t *handle_out);
extern uint32_t hmcuda_core_cu_multicast_add_device(uint32_t sid, uint32_t vid, uint64_t mc_handle_guest, int32_t device);
extern uint32_t hmcuda_core_cu_multicast_bind_mem(uint32_t sid, uint32_t vid, uint64_t mc_handle_guest, uint64_t mc_offset, uint64_t mem_handle_guest, uint64_t mem_offset, uint64_t size, uint64_t flags);
extern uint32_t hmcuda_core_cu_multicast_unbind(uint32_t sid, uint32_t vid, uint64_t mc_handle_guest, int32_t device, uint64_t mc_offset, uint64_t size);

static size_t buf_to_iov(const struct iovec *iov, unsigned int iov_cnt, const void *buf, size_t len)
{
    size_t offset = 0;
    unsigned int i;
    for (i = 0; i < iov_cnt && offset < len; i++) {
        size_t copy = len - offset;
        if (copy > iov[i].iov_len)
            copy = iov[i].iov_len;
        memcpy(iov[i].iov_base, (const char *)buf + offset, copy);
        offset += copy;
    }
    return offset;
}

/*
 * Driver API page-pinning memcpy: same virtqueue layout as HMCUDA_CMD_MEMCPY
 * but calls cuMemcpyHtoD / cuMemcpyDtoH instead of the Runtime API.
 */
static void handle_cu_memcpy(struct hmcuda_vhost_cmd_ctx *ctx)
{
    struct hmcuda_memcpy_req *req = ctx->payload;
    uint32_t sid = ctx->session_id;
    uint32_t vid = ctx->vm_id;
    VuVirtqElement *elem = ctx->elem;

    LOG_DEBUG("[CU_MEMCPY] kind=%u count=%lu src=0x%lx dst=0x%lx out_num=%u in_num=%u",
            req->kind, req->count, req->src, req->dst, elem->out_num, elem->in_num);

    if (req->kind == 1) { /* HostToDevice */
        if (elem->out_num >= 2) {
            size_t req_struct_len = sizeof(struct hmcuda_req_header) + sizeof(*req);
            size_t avail = ctx->total_out_len - req_struct_len;
            if (avail >= req->count) {
                size_t offset = 0;
                ctx->cuda_error = 0;
                for (unsigned i = 1; i < elem->out_num && offset < req->count; i++) {
                    size_t chunk = elem->out_sg[i].iov_len;
                    if (offset + chunk > req->count) chunk = req->count - offset;
                    ctx->cuda_error = hmcuda_core_cu_memcpy_htod_vq(sid, vid, req->dst + offset, elem->out_sg[i].iov_base, chunk, req->stream);
                    if (ctx->cuda_error != 0) break;
                    offset += chunk;
                }
            } else {
                LOG_ERROR("[CU_MEMCPY] H2D: not enough data (%zu < %lu)", avail, req->count);
                ctx->cuda_error = 1;
            }
        } else {
            LOG_ERROR("[CU_MEMCPY] H2D: not enough out buffers");
            ctx->cuda_error = 1;
        }
    } else if (req->kind == 2) { /* DeviceToHost */
        ctx->d2h_virtqueue = 1;
        if (elem->in_num >= 2) {
            size_t data_capacity = 0;
            for (unsigned i = 0; i < elem->in_num - 1; i++)
                data_capacity += elem->in_sg[i].iov_len;
            if (data_capacity >= req->count) {
                /* Guest RAM is registered with CUDA via cuMemHostRegister at context
                 * creation, so iov_base is GPU-DMA-accessible.  Submit all segments
                 * async and return immediately — no per-batch sync.  Data becomes
                 * valid after the guest calls cuStreamSynchronize / cuEventSynchronize,
                 * which goes through hmcuda_core_cu_stream_synchronize on the host. */
                size_t offset = 0;
                ctx->cuda_error = 0;
                for (unsigned i = 0; i < elem->in_num - 1 && offset < req->count; i++) {
                    size_t chunk = elem->in_sg[i].iov_len;
                    if (offset + chunk > req->count) chunk = req->count - offset;
                    ctx->cuda_error = hmcuda_core_cu_memcpy_dtoh_vq(
                        sid, vid, req->src + offset,
                        elem->in_sg[i].iov_base, chunk, req->stream);
                    if (ctx->cuda_error != 0) break;
                    offset += chunk;
                }
            } else {
                LOG_ERROR("[CU_MEMCPY] D2H: buffer too small (%zu < %lu)", data_capacity, req->count);
                ctx->cuda_error = 1;
            }
        } else {
            LOG_ERROR("[CU_MEMCPY] D2H: not enough in buffers");
            ctx->cuda_error = 1;
        }
    } else {
        LOG_ERROR("[CU_MEMCPY] unexpected kind=%u", req->kind);
        ctx->cuda_error = 1;
    }
}

/*
 * cuMemHostAlloc page-pin path: same virtqueue layout as handle_cu_memcpy H2D.
 * Page data arrives inline after the req struct; backend allocates real_ptr on
 * first batch (byte_offset==0) then memcpy's each batch into real_ptr+byte_offset.
 */
static void handle_cu_mem_host_alloc(struct hmcuda_vhost_cmd_ctx *ctx)
{
    struct hmcuda_cu_mem_host_alloc_req *req = ctx->payload;
    uint32_t sid = ctx->session_id;
    uint32_t vid = ctx->vm_id;
    VuVirtqElement *elem = ctx->elem;

    if (elem->out_num >= 2) {
        size_t req_struct_len = sizeof(struct hmcuda_req_header) + sizeof(*req);
        size_t batch_size = ctx->total_out_len - req_struct_len;
        size_t offset = 0;
        ctx->cuda_error = 0;
        for (unsigned i = 1; i < elem->out_num && offset < batch_size; i++) {
            size_t chunk = elem->out_sg[i].iov_len;
            ctx->cuda_error = hmcuda_core_cu_mem_host_alloc_vq(
                sid, vid, req->bytesize, req->flags, req->guest_va,
                req->byte_offset + offset, elem->out_sg[i].iov_base, chunk);
            if (ctx->cuda_error != 0) break;
            offset += chunk;
        }
    } else {
        LOG_ERROR("[CU_MEM_HOST_ALLOC] missing page data in virtqueue");
        ctx->cuda_error = 1;
    }
}

/*
 * Handle a CUDA Driver API command.
 * Returns 0 if handled, -1 if the command is not in the driver range.
 */
int hmcuda_driver_handle_cmd(uint32_t cmd, struct hmcuda_vhost_cmd_ctx *ctx)
{
    uint32_t sid = ctx->session_id;
    uint32_t vid = ctx->vm_id;
    void *payload = ctx->payload;

    switch (cmd) {

    /* ---- Initialization & version (stateless) ---- */

    case HMCUDA_CMD_CU_INIT: {
        struct hmcuda_cu_init_req *req = payload;
        ctx->cuda_error = cuInit(req->flags);
        break;
    }
    case HMCUDA_CMD_CU_DRIVER_GET_VERSION: {
        struct hmcuda_cu_driver_get_version_resp resp;
        ctx->cuda_error = cuDriverGetVersion(&resp.version);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }

    /* ---- Device management (stateless) ---- */

    case HMCUDA_CMD_CU_DEVICE_GET: {
        struct hmcuda_cu_device_get_req *req = payload;
        struct hmcuda_cu_device_get_resp resp;
        CUdevice dev;
        ctx->cuda_error = cuDeviceGet(&dev, req->ordinal);
        resp.device = (int32_t)dev;
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_DEVICE_GET_COUNT: {
        struct hmcuda_cu_device_get_count_resp resp;
        ctx->cuda_error = cuDeviceGetCount(&resp.count);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_DEVICE_GET_NAME: {
        struct hmcuda_cu_device_get_name_req *req = payload;
        struct hmcuda_cu_device_get_name_resp resp;
        int max_len = req->name_len > 0 ? req->name_len : 256;
        char *name = calloc(1, max_len);
        if (!name) { ctx->cuda_error = CUDA_ERROR_OUT_OF_MEMORY; break; }
        ctx->cuda_error = cuDeviceGetName(name, max_len, (CUdevice)req->device);
        int actual_len = (int)strlen(name) + 1;
        resp.name_len = actual_len;
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        memcpy(ctx->resp_payload + sizeof(resp), name, actual_len);
        ctx->resp_payload_len = sizeof(resp) + actual_len;
        free(name);
        break;
    }
    case HMCUDA_CMD_CU_DEVICE_GET_ATTRIBUTE: {
        struct hmcuda_cu_device_get_attribute_req *req = payload;
        struct hmcuda_cu_device_get_attribute_resp resp;
        ctx->cuda_error = cuDeviceGetAttribute(&resp.value,
            (CUdevice_attribute)req->attrib, (CUdevice)req->device);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_DEVICE_GET_UUID: {
        struct hmcuda_cu_device_get_uuid_req *req = payload;
        struct hmcuda_cu_device_get_uuid_resp resp;
        CUuuid uuid;
        ctx->cuda_error = cuDeviceGetUuid(&uuid, (CUdevice)req->device);
        memcpy(resp.uuid, uuid.bytes, 16);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_DEVICE_CAN_ACCESS_PEER: {
        struct hmcuda_cu_device_can_access_peer_req *req = payload;
        struct hmcuda_cu_device_can_access_peer_resp resp;
        ctx->cuda_error = cuDeviceCanAccessPeer(&resp.can_access,
            (CUdevice)req->device, (CUdevice)req->peer_device);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }

    /* ---- Device management (resource-managed) ---- */

    case HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RETAIN: {
        struct hmcuda_cu_device_primary_ctx_retain_req *req = payload;
        struct hmcuda_cu_device_primary_ctx_retain_resp resp;
        ctx->cuda_error = hmcuda_core_cu_device_primary_ctx_retain(sid, vid, req->device, &resp.context);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RELEASE: {
        struct hmcuda_cu_device_primary_ctx_release_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_device_primary_ctx_release(sid, vid, req->device);
        break;
    }

    /* ---- Context management ---- */

    case HMCUDA_CMD_CU_CTX_SET_CURRENT: {
        struct hmcuda_cu_ctx_set_current_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_ctx_set_current(sid, vid, req->context);
        break;
    }
    case HMCUDA_CMD_CU_CTX_GET_DEVICE: {
        struct hmcuda_cu_ctx_get_device_resp resp;
        CUdevice dev;
        ctx->cuda_error = cuCtxGetDevice(&dev);
        resp.device = (int32_t)dev;
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_CTX_ENABLE_PEER_ACCESS: {
        struct hmcuda_cu_ctx_enable_peer_access_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_ctx_enable_peer_access(sid, vid, req->peer_context, req->flags);
        break;
    }

    /* ---- Stream management ---- */

    case HMCUDA_CMD_CU_STREAM_CREATE: {
        struct hmcuda_cu_stream_create_req *req = payload;
        struct hmcuda_cu_stream_create_resp resp;
        ctx->cuda_error = hmcuda_core_cu_stream_create(sid, vid, req->flags, &resp.stream);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_STREAM_DESTROY: {
        struct hmcuda_cu_stream_destroy_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_stream_destroy(sid, vid, req->stream);
        break;
    }
    case HMCUDA_CMD_CU_STREAM_SYNCHRONIZE: {
        struct hmcuda_cu_stream_synchronize_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_stream_synchronize(sid, vid, req->stream);
        break;
    }
    case HMCUDA_CMD_CU_STREAM_QUERY: {
        struct hmcuda_cu_stream_query_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_stream_query(sid, vid, req->stream);
        break;
    }
    case HMCUDA_CMD_CU_STREAM_WAIT_EVENT: {
        struct hmcuda_cu_stream_wait_event_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_stream_wait_event(sid, vid, req->stream, req->event, req->flags);
        break;
    }
    case HMCUDA_CMD_CU_STREAM_GET_CTX: {
        struct hmcuda_cu_stream_get_ctx_req *req = payload;
        struct hmcuda_cu_stream_get_ctx_resp resp;
        ctx->cuda_error = hmcuda_core_cu_stream_get_ctx(sid, vid, req->stream, &resp.context);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }

    /* ---- Event management ---- */

    case HMCUDA_CMD_CU_EVENT_CREATE: {
        struct hmcuda_cu_event_create_req *req = payload;
        struct hmcuda_cu_event_create_resp resp;
        ctx->cuda_error = hmcuda_core_cu_event_create(sid, vid, req->flags, &resp.event);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_EVENT_DESTROY: {
        struct hmcuda_cu_event_destroy_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_event_destroy(sid, vid, req->event);
        break;
    }
    case HMCUDA_CMD_CU_EVENT_RECORD: {
        struct hmcuda_cu_event_record_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_event_record(sid, vid, req->event, req->stream);
        break;
    }
    case HMCUDA_CMD_CU_EVENT_ELAPSED_TIME: {
        struct hmcuda_cu_event_elapsed_time_req *req = payload;
        struct hmcuda_cu_event_elapsed_time_resp resp;
        ctx->cuda_error = hmcuda_core_cu_event_elapsed_time(sid, vid, req->start_event, req->end_event, &resp.ms);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }

    /* ---- Memory allocation ---- */

    case HMCUDA_CMD_CU_MEM_ALLOC: {
        struct hmcuda_cu_mem_alloc_req *req = payload;
        struct hmcuda_cu_mem_alloc_resp resp;
        ctx->cuda_error = hmcuda_core_cu_mem_alloc(sid, vid, req->bytesize, &resp.dptr);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_MEM_FREE: {
        struct hmcuda_cu_mem_free_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_mem_free(sid, vid, req->dptr);
        break;
    }
    case HMCUDA_CMD_CU_MEM_HOST_ALLOC:
        handle_cu_mem_host_alloc(ctx);
        break;
    case HMCUDA_CMD_CU_MEM_FREE_HOST: {
        struct hmcuda_cu_mem_free_host_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_mem_free_host(sid, vid, req->guest_va);
        break;
    }
    case HMCUDA_CMD_CU_MEMSET_D32: {
        struct hmcuda_cu_memset_d32_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_memset_d32(sid, vid, req->dptr, req->value, req->count);
        break;
    }

    /* ---- Memory copy ---- */

    case HMCUDA_CMD_CU_MEMCPY_ASYNC: {
        struct hmcuda_cu_memcpy_async_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_memcpy_async(sid, vid, req->dst, req->src, req->bytesize, req->stream);
        break;
    }
    case HMCUDA_CMD_CU_MEMCPY_HTOD: {
        struct hmcuda_cu_memcpy_htod_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_memcpy_htod(sid, vid, req->dst_device, req->src_host, req->bytesize);
        break;
    }
    case HMCUDA_CMD_CU_MEMCPY:
        handle_cu_memcpy(ctx);
        break;

    /* ---- Pointer & error ---- */

    case HMCUDA_CMD_CU_POINTER_GET_ATTRIBUTE: {
        struct hmcuda_cu_pointer_get_attribute_req *req = payload;
        struct hmcuda_cu_pointer_get_attribute_resp resp;
        ctx->cuda_error = hmcuda_core_cu_pointer_get_attribute(sid, vid, req->attrib, req->ptr, &resp.value);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_GET_ERROR_STRING: {
        struct hmcuda_cu_get_error_string_req *req = payload;
        struct hmcuda_cu_get_error_string_resp resp;
        const char *str = NULL;
        ctx->cuda_error = cuGetErrorString((CUresult)req->error, &str);
        if (str) {
            uint32_t len = (uint32_t)strlen(str) + 1;
            resp.str_len = len;
            memcpy(ctx->resp_payload, &resp, sizeof(resp));
            memcpy(ctx->resp_payload + sizeof(resp), str, len);
            ctx->resp_payload_len = sizeof(resp) + len;
        } else {
            resp.str_len = 0;
            memcpy(ctx->resp_payload, &resp, sizeof(resp));
            ctx->resp_payload_len = sizeof(resp);
        }
        break;
    }
    case HMCUDA_CMD_CU_GET_ERROR_NAME: {
        struct hmcuda_cu_get_error_name_req *req = payload;
        struct hmcuda_cu_get_error_name_resp resp;
        const char *str = NULL;
        ctx->cuda_error = cuGetErrorName((CUresult)req->error, &str);
        if (str) {
            uint32_t len = (uint32_t)strlen(str) + 1;
            resp.str_len = len;
            memcpy(ctx->resp_payload, &resp, sizeof(resp));
            memcpy(ctx->resp_payload + sizeof(resp), str, len);
            ctx->resp_payload_len = sizeof(resp) + len;
        } else {
            resp.str_len = 0;
            memcpy(ctx->resp_payload, &resp, sizeof(resp));
            ctx->resp_payload_len = sizeof(resp);
        }
        break;
    }

    /* ---- Virtual memory management ---- */

    case HMCUDA_CMD_CU_MEM_CREATE: {
        struct hmcuda_cu_mem_create_req *req = payload;
        struct hmcuda_cu_mem_create_resp resp;
        CUmemAllocationProp prop = {0};
        prop.type = (CUmemAllocationType)req->prop.type;
        prop.location.type = (CUmemLocationType)req->prop.location_type;
        prop.location.id = req->prop.location_id;
        prop.requestedHandleTypes = (CUmemAllocationHandleType)req->prop.requested_handle_types;
        ctx->cuda_error = hmcuda_core_cu_mem_create(sid, vid, req->size, &prop, req->flags, &resp.handle);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_MEM_RELEASE: {
        struct hmcuda_cu_mem_release_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_mem_release(sid, vid, req->handle);
        break;
    }
    case HMCUDA_CMD_CU_MEM_MAP: {
        struct hmcuda_cu_mem_map_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_mem_map(sid, vid, req->ptr, req->size, req->offset, req->handle, req->flags);
        break;
    }
    case HMCUDA_CMD_CU_MEM_UNMAP: {
        struct hmcuda_cu_mem_unmap_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_mem_unmap(sid, vid, req->ptr, req->size);
        break;
    }
    case HMCUDA_CMD_CU_MEM_ADDRESS_RESERVE: {
        struct hmcuda_cu_mem_address_reserve_req *req = payload;
        struct hmcuda_cu_mem_address_reserve_resp resp;
        ctx->cuda_error = hmcuda_core_cu_mem_address_reserve(sid, vid, req->size, req->alignment, req->addr, req->flags, &resp.ptr);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_MEM_ADDRESS_FREE: {
        struct hmcuda_cu_mem_address_free_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_mem_address_free(sid, vid, req->ptr, req->size);
        break;
    }
    case HMCUDA_CMD_CU_MEM_SET_ACCESS: {
        struct hmcuda_cu_mem_set_access_req *req = payload;
        struct hmcuda_cu_mem_access_desc *descs =
            (struct hmcuda_cu_mem_access_desc *)((char *)payload + sizeof(*req));
        CUmemAccessDesc *cu_descs = calloc(req->count, sizeof(CUmemAccessDesc));
        if (!cu_descs) { ctx->cuda_error = CUDA_ERROR_OUT_OF_MEMORY; break; }
        for (uint32_t i = 0; i < req->count; i++) {
            cu_descs[i].location.type = (CUmemLocationType)descs[i].location_type;
            cu_descs[i].location.id = descs[i].location_id;
            cu_descs[i].flags = (CUmemAccess_flags)descs[i].flags;
        }
        ctx->cuda_error = hmcuda_core_cu_mem_set_access(sid, vid, req->ptr, req->size, cu_descs, (size_t)req->count);
        free(cu_descs);
        break;
    }
    case HMCUDA_CMD_CU_MEM_GET_ALLOC_GRANULARITY: {
        struct hmcuda_cu_mem_get_alloc_granularity_req *req = payload;
        struct hmcuda_cu_mem_get_alloc_granularity_resp resp;
        CUmemAllocationProp prop = {0};
        prop.type = (CUmemAllocationType)req->prop.type;
        prop.location.type = (CUmemLocationType)req->prop.location_type;
        prop.location.id = req->prop.location_id;
        prop.requestedHandleTypes = (CUmemAllocationHandleType)req->prop.requested_handle_types;
        size_t gran;
        ctx->cuda_error = cuMemGetAllocationGranularity(
            &gran, &prop, (CUmemAllocationGranularity_flags)req->option);
        resp.granularity = (uint64_t)gran;
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_MEM_EXPORT_TO_SHAREABLE_HANDLE: {
        struct hmcuda_cu_mem_export_req *req = payload;
        struct hmcuda_cu_mem_export_resp resp;
        memset(&resp, 0, sizeof(resp));
        ctx->cuda_error = hmcuda_core_cu_mem_export(sid, vid, req->handle, req->handle_type, req->flags, resp.shareable_handle);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_MEM_IMPORT_FROM_SHAREABLE_HANDLE: {
        struct hmcuda_cu_mem_import_req *req = payload;
        struct hmcuda_cu_mem_import_resp resp;
        ctx->cuda_error = hmcuda_core_cu_mem_import(sid, vid, req->shareable_handle, req->handle_type, &resp.handle);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }

    /* ---- Multicast memory ---- */

    case HMCUDA_CMD_CU_MULTICAST_CREATE: {
        struct hmcuda_cu_multicast_create_req *req = payload;
        struct hmcuda_cu_multicast_create_resp resp;
        CUmulticastObjectProp mc_prop = {0};
        mc_prop.numDevices = req->prop.num_devices;
        mc_prop.handleTypes = (CUmemAllocationHandleType)req->prop.handle_types;
        mc_prop.size = (size_t)req->prop.size;
        ctx->cuda_error = hmcuda_core_cu_multicast_create(sid, vid, &mc_prop, &resp.mc_handle);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_MULTICAST_GET_GRANULARITY: {
        struct hmcuda_cu_multicast_get_granularity_req *req = payload;
        struct hmcuda_cu_multicast_get_granularity_resp resp;
        CUmulticastObjectProp mc_prop = {0};
        mc_prop.numDevices = req->prop.num_devices;
        mc_prop.handleTypes = (CUmemAllocationHandleType)req->prop.handle_types;
        mc_prop.size = (size_t)req->prop.size;
        size_t gran;
        ctx->cuda_error = cuMulticastGetGranularity(
            &gran, &mc_prop, (CUmulticastGranularity_flags)req->option);
        resp.granularity = (uint64_t)gran;
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_CU_MULTICAST_ADD_DEVICE: {
        struct hmcuda_cu_multicast_add_device_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_multicast_add_device(sid, vid, req->mc_handle, req->device);
        break;
    }
    case HMCUDA_CMD_CU_MULTICAST_BIND_MEM: {
        struct hmcuda_cu_multicast_bind_mem_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_multicast_bind_mem(sid, vid,
            req->mc_handle, req->mc_offset, req->mem_handle, req->mem_offset,
            req->size, req->flags);
        break;
    }
    case HMCUDA_CMD_CU_MULTICAST_UNBIND: {
        struct hmcuda_cu_multicast_unbind_req *req = payload;
        ctx->cuda_error = hmcuda_core_cu_multicast_unbind(sid, vid,
            req->mc_handle, req->device, req->mc_offset, req->size);
        break;
    }

    default:
        return -1; /* Not a driver API command */
    }
    return 0;
}
