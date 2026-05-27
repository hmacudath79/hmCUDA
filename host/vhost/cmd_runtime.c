#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/uio.h>

#include "hmcuda_api.h"
#include "cmd_dispatch.h"
#include "log.h"

/* Core API prototypes (implemented in core.c) */
extern uint32_t hmcuda_core_init(uint32_t session_id, uint32_t vm_id);
extern uint32_t hmcuda_core_malloc(uint32_t session_id, uint32_t vm_id, uint64_t size, uint64_t *devPtr);
extern uint32_t hmcuda_core_free(uint32_t session_id, uint32_t vm_id, uint64_t devPtr);
extern uint32_t hmcuda_core_memcpy(uint32_t session_id, uint32_t vm_id, uint64_t dst, uint64_t src, uint64_t count, uint32_t kind, void *host_ptr, uint64_t stream_handle);
extern uint32_t hmcuda_core_runtime_memcpy_htod_vq(uint32_t session_id, uint32_t vm_id, uint64_t dst, void *host_ptr, uint64_t count, uint64_t stream_handle);
extern uint32_t hmcuda_core_memset(uint32_t session_id, uint32_t vm_id, uint64_t devPtr, int32_t value, uint64_t count);
extern uint32_t hmcuda_core_launch_kernel(uint32_t session_id, uint32_t vm_id, uint64_t func, struct hmcuda_dim3 gridDim, struct hmcuda_dim3 blockDim, uint32_t sharedMem, uint64_t stream, void *args, uint32_t args_size);
extern uint32_t hmcuda_core_stream_create(uint32_t session_id, uint32_t vm_id, uint32_t flags, uint64_t *stream);
extern uint32_t hmcuda_core_stream_synchronize(uint32_t session_id, uint32_t vm_id, uint64_t stream);
extern uint32_t hmcuda_core_event_create(uint32_t session_id, uint32_t vm_id, uint32_t flags, uint64_t *event);
extern uint32_t hmcuda_core_event_record(uint32_t session_id, uint32_t vm_id, uint64_t event, uint64_t stream);
extern uint32_t hmcuda_core_event_synchronize(uint32_t session_id, uint32_t vm_id, uint64_t event);
extern uint32_t hmcuda_core_event_elapsed_time(uint32_t session_id, uint32_t vm_id, uint64_t start, uint64_t end, float *ms);
extern uint32_t hmcuda_core_event_destroy(uint32_t session_id, uint32_t vm_id, uint64_t event);
extern uint32_t hmcuda_core_device_synchronize(uint32_t session_id, uint32_t vm_id);
extern uint32_t hmcuda_core_push_call_configuration(uint32_t session_id, uint32_t vm_id, struct hmcuda_dim3 gridDim, struct hmcuda_dim3 blockDim, uint64_t sharedMem, uint64_t stream);
extern uint32_t hmcuda_core_pop_call_configuration(uint32_t session_id, uint32_t vm_id, struct hmcuda_dim3 *gridDim, struct hmcuda_dim3 *blockDim, uint64_t *sharedMem, uint64_t *stream);
extern uint32_t hmcuda_core_register_fatbin(uint32_t session_id, uint32_t vm_id, void *fatbin_data, uint64_t size, uint64_t *fatbin_handle);
extern uint32_t hmcuda_core_register_function(uint32_t session_id, uint32_t vm_id, uint64_t fatbin_handle, uint64_t host_fun, const char *device_name, uint32_t *out_param_count, uint32_t *out_param_total_size);
extern uint32_t hmcuda_core_init_module(uint32_t session_id, uint32_t vm_id, uint64_t fatbin_handle);
extern uint32_t hmcuda_core_get_device_properties(uint32_t session_id, uint32_t vm_id, int device, struct hmcuda_device_prop *out);
extern uint32_t hmcuda_core_get_device_count(uint32_t session_id, uint32_t vm_id, int *count);
extern uint32_t hmcuda_core_set_device(uint32_t session_id, uint32_t vm_id, int device);
extern uint32_t hmcuda_core_get_error_string(uint32_t error, const char **str);
extern uint32_t hmcuda_core_get_error_name(uint32_t error, const char **str);
extern uint32_t hmcuda_core_runtime_get_version(int *version);
extern uint32_t hmcuda_core_func_get_attributes(uint32_t session_id, uint32_t vm_id, uint64_t func_handle, struct hmcuda_func_attributes *out);

static size_t buf_to_iov(const struct iovec *iov, unsigned int iov_cnt, const void *buf, size_t len)
{
    size_t offset = 0;
    unsigned int i;
    for (i = 0; i < iov_cnt && offset < len; i++) {
        size_t copy = len - offset;
        if (copy > iov[i].iov_len)
            copy = iov[i].iov_len;
        memcpy(iov[i].iov_base, (char *)buf + offset, copy);
        offset += copy;
    }
    return offset;
}

static void handle_memcpy(struct hmcuda_vhost_cmd_ctx *ctx)
{
    struct hmcuda_memcpy_req *req = ctx->payload;
    void *host_ptr = NULL;
    uint32_t sid = ctx->session_id;
    uint32_t vid = ctx->vm_id;
    VuDev *dev = ctx->dev;
    VuVirtqElement *elem = ctx->elem;

    LOG_DEBUG("[MEMCPY] kind=%u count=%lu src=0x%lx dst=0x%lx out_num=%u in_num=%u",
            req->kind, req->count, req->src, req->dst, elem->out_num, elem->in_num);

    if (req->kind == 1) { /* HostToDevice */
        if (req->src == 0) {
            LOG_DEBUG("[MEMCPY] H2D: Data from virtqueue");
            if (elem->out_num >= 2) {
                size_t req_struct_len = sizeof(struct hmcuda_req_header) + sizeof(*req);
                size_t avail = ctx->total_out_len - req_struct_len;
                LOG_DEBUG("[MEMCPY] H2D: out_num=%u, avail=%zu, req->count=%lu",
                        elem->out_num, avail, req->count);
                if (avail >= req->count) {
                    size_t offset = 0;
                    ctx->cuda_error = 0;
                    for (unsigned i = 1; i < elem->out_num && offset < req->count; i++) {
                        size_t chunk = elem->out_sg[i].iov_len;
                        if (offset + chunk > req->count) chunk = req->count - offset;
                        ctx->cuda_error = hmcuda_core_runtime_memcpy_htod_vq(sid, vid, req->dst + offset, elem->out_sg[i].iov_base, chunk, req->stream);
                        if (ctx->cuda_error != 0) break;
                        offset += chunk;
                    }
                    LOG_DEBUG("[MEMCPY] H2D: cuMemcpy returned %u", ctx->cuda_error);
                } else {
                    LOG_DEBUG("[MEMCPY] H2D: ERROR - Not enough data (%zu < %lu)", avail, req->count);
                    ctx->cuda_error = 1;
                }
            } else {
                LOG_DEBUG("[MEMCPY] H2D: ERROR - Not enough out buffers");
                ctx->cuda_error = 1;
            }
        } else {
            LOG_DEBUG("[MEMCPY] H2D: Legacy mode (GPA)");
            uint64_t plen = req->count;
            host_ptr = vu_gpa_to_va(dev, &plen, req->src);
            if (!host_ptr || plen < req->count) {
                LOG_DEBUG("[MEMCPY] H2D: GPA translation failed");
                ctx->cuda_error = 1;
            } else {
                ctx->cuda_error = hmcuda_core_memcpy(sid, vid, req->dst, 0, req->count, req->kind, host_ptr, req->stream);
            }
        }
    } else if (req->kind == 2) { /* DeviceToHost */
        LOG_DEBUG("[MEMCPY] D2H: Processing");
        if (req->dst == 0) {
            LOG_DEBUG("[MEMCPY] D2H: Data to virtqueue");
            ctx->d2h_virtqueue = 1;
            if (elem->in_num >= 2) {
                size_t data_capacity = 0;
                for (unsigned i = 0; i < elem->in_num - 1; i++)
                    data_capacity += elem->in_sg[i].iov_len;
                LOG_DEBUG("[MEMCPY] D2H: in_num=%u, data_capacity=%zu, req->count=%lu",
                        elem->in_num, data_capacity, req->count);
                if (data_capacity >= req->count) {
                    size_t offset = 0;
                    ctx->cuda_error = 0;
                    for (unsigned i = 0; i < elem->in_num - 1 && offset < req->count; i++) {
                        size_t chunk = elem->in_sg[i].iov_len;
                        if (offset + chunk > req->count) chunk = req->count - offset;
                        ctx->cuda_error = hmcuda_core_memcpy(sid, vid, 0, req->src + offset, chunk, req->kind, elem->in_sg[i].iov_base, req->stream);
                        LOG_DEBUG("[MEMCPY] D2H: chunk %u returned %u", i, ctx->cuda_error);
                        if (ctx->cuda_error != 0) break;
                        offset += chunk;
                    }
                    if (ctx->cuda_error == 0)
                        ctx->cuda_error = hmcuda_core_stream_synchronize(sid, vid, req->stream);
                } else {
                    LOG_DEBUG("[MEMCPY] D2H: ERROR - Buffer too small (%zu < %lu)", data_capacity, req->count);
                    ctx->cuda_error = 1;
                }
            } else {
                LOG_DEBUG("[MEMCPY] D2H: ERROR - Not enough in buffers");
                ctx->cuda_error = 1;
            }
        } else {
            LOG_DEBUG("[MEMCPY] D2H: Legacy mode (GPA)");
            uint64_t plen = req->count;
            host_ptr = vu_gpa_to_va(dev, &plen, req->dst);
            if (!host_ptr || plen < req->count) {
                LOG_DEBUG("[MEMCPY] D2H: GPA translation failed");
                ctx->cuda_error = 1;
            } else {
                ctx->cuda_error = hmcuda_core_memcpy(sid, vid, 0, req->src, req->count, req->kind, host_ptr, req->stream);
                if (ctx->cuda_error == 0)
                    ctx->cuda_error = hmcuda_core_stream_synchronize(sid, vid, req->stream);
            }
        }
    } else { /* DeviceToDevice */
        LOG_DEBUG("[MEMCPY] D2D: Processing");
        ctx->cuda_error = hmcuda_core_memcpy(sid, vid, req->dst, req->src, req->count, req->kind, NULL, req->stream);
    }
}

/*
 * Handle a CUDA Runtime API command.
 * Returns 0 if handled, -1 if the command is not in the runtime range.
 */
int hmcuda_runtime_handle_cmd(uint32_t cmd, struct hmcuda_vhost_cmd_ctx *ctx)
{
    uint32_t sid = ctx->session_id;
    uint32_t vid = ctx->vm_id;
    void *payload = ctx->payload;

    switch (cmd) {
    case HMCUDA_CMD_INIT:
        ctx->cuda_error = hmcuda_core_init(sid, vid);
        break;
    case HMCUDA_CMD_MALLOC: {
        struct hmcuda_malloc_req *req = payload;
        struct hmcuda_malloc_resp resp;
        ctx->cuda_error = hmcuda_core_malloc(sid, vid, req->size, &resp.devPtr);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_FREE: {
        struct hmcuda_free_req *req = payload;
        ctx->cuda_error = hmcuda_core_free(sid, vid, req->devPtr);
        break;
    }
    case HMCUDA_CMD_MEMCPY:
        handle_memcpy(ctx);
        break;
    case HMCUDA_CMD_MEMSET: {
        struct hmcuda_memset_req *req = payload;
        ctx->cuda_error = hmcuda_core_memset(sid, vid, req->devPtr, req->value, req->count);
        break;
    }
    case HMCUDA_CMD_LAUNCH_KERNEL: {
        struct hmcuda_launch_kernel_req *req = payload;
        void *args = (char *)payload + sizeof(*req);
        ctx->cuda_error = hmcuda_core_launch_kernel(sid, vid, req->func, req->gridDim, req->blockDim, req->sharedMem, req->stream, args, req->args_size);
        break;
    }
    case HMCUDA_CMD_STREAM_CREATE: {
        struct hmcuda_stream_create_req *req = payload;
        struct hmcuda_stream_create_resp resp;
        ctx->cuda_error = hmcuda_core_stream_create(sid, vid, req->flags, &resp.stream);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_STREAM_SYNCHRONIZE: {
        struct hmcuda_stream_synchronize_req *req = payload;
        ctx->cuda_error = hmcuda_core_stream_synchronize(sid, vid, req->stream);
        break;
    }
    case HMCUDA_CMD_EVENT_CREATE: {
        struct hmcuda_event_create_req *req = payload;
        struct hmcuda_event_create_resp resp;
        ctx->cuda_error = hmcuda_core_event_create(sid, vid, req->flags, &resp.event);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_EVENT_RECORD: {
        struct hmcuda_event_record_req *req = payload;
        ctx->cuda_error = hmcuda_core_event_record(sid, vid, req->event, req->stream);
        break;
    }
    case HMCUDA_CMD_EVENT_SYNCHRONIZE: {
        struct hmcuda_event_synchronize_req *req = payload;
        ctx->cuda_error = hmcuda_core_event_synchronize(sid, vid, req->event);
        break;
    }
    case HMCUDA_CMD_EVENT_ELAPSED_TIME: {
        struct hmcuda_event_elapsed_time_req *req = payload;
        struct hmcuda_event_elapsed_time_resp resp;
        ctx->cuda_error = hmcuda_core_event_elapsed_time(sid, vid, req->start_event, req->end_event, &resp.ms);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_EVENT_DESTROY: {
        struct hmcuda_event_destroy_req *req = payload;
        ctx->cuda_error = hmcuda_core_event_destroy(sid, vid, req->event);
        break;
    }
    case HMCUDA_CMD_DEVICE_SYNCHRONIZE:
        ctx->cuda_error = hmcuda_core_device_synchronize(sid, vid);
        break;
    case HMCUDA_CMD_PUSH_CALL_CONFIGURATION: {
        struct hmcuda_push_call_cfg_req *req = payload;
        ctx->cuda_error = hmcuda_core_push_call_configuration(sid, vid, req->gridDim, req->blockDim, req->sharedMem, req->stream);
        break;
    }
    case HMCUDA_CMD_POP_CALL_CONFIGURATION: {
        struct hmcuda_pop_call_cfg_resp resp;
        ctx->cuda_error = hmcuda_core_pop_call_configuration(sid, vid, &resp.gridDim, &resp.blockDim, &resp.sharedMem, &resp.stream);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_INIT_MODULE: {
        struct hmcuda_init_module_req *req = payload;
        ctx->cuda_error = hmcuda_core_init_module(sid, vid, req->fatbin_handle);
        break;
    }
    case HMCUDA_CMD_GET_DEVICE_PROPERTIES: {
        struct hmcuda_get_device_properties_req *req = payload;
        struct hmcuda_get_device_properties_resp resp;
        ctx->cuda_error = hmcuda_core_get_device_properties(sid, vid, req->device, &resp.prop);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_GET_DEVICE_COUNT: {
        struct hmcuda_get_device_count_resp resp;
        ctx->cuda_error = hmcuda_core_get_device_count(sid, vid, &resp.count);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_SET_DEVICE: {
        struct hmcuda_set_device_req *req = payload;
        ctx->cuda_error = hmcuda_core_set_device(sid, vid, req->device);
        break;
    }
    case HMCUDA_CMD_GET_ERROR_STRING:
    case HMCUDA_CMD_GET_ERROR_NAME: {
        struct hmcuda_get_error_string_req *req = payload;
        const char *str = NULL;
        if (cmd == HMCUDA_CMD_GET_ERROR_STRING)
            ctx->cuda_error = hmcuda_core_get_error_string(req->error, &str);
        else
            ctx->cuda_error = hmcuda_core_get_error_name(req->error, &str);
        if (str) {
            size_t slen = strlen(str) + 1;
            struct hmcuda_get_error_string_resp resp = { .str_len = (uint32_t)slen };
            memcpy(ctx->resp_payload, &resp, sizeof(resp));
            if (slen + sizeof(resp) <= ctx->resp_payload_cap)
                memcpy((char *)ctx->resp_payload + sizeof(resp), str, slen);
            ctx->resp_payload_len = sizeof(resp) + slen;
        }
        break;
    }
    case HMCUDA_CMD_REGISTER_FATBIN: {
        struct hmcuda_register_fatbin_req *req = payload;
        void *fatbin_data = (char *)payload + sizeof(*req);
        struct hmcuda_register_fatbin_resp resp;
        ctx->cuda_error = hmcuda_core_register_fatbin(sid, vid, fatbin_data, req->size, &resp.fatbin_handle);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_REGISTER_FUNCTION: {
        struct hmcuda_register_function_req *req = payload;
        const char *device_name = (char *)payload + sizeof(*req);
        struct hmcuda_register_function_resp func_resp;
        ctx->cuda_error = hmcuda_core_register_function(sid, vid, req->fatbin_handle, req->host_fun, device_name, &func_resp.param_count, &func_resp.param_total_size);
        memcpy(ctx->resp_payload, &func_resp, sizeof(func_resp));
        ctx->resp_payload_len = sizeof(func_resp);
        break;
    }
    case HMCUDA_CMD_RUNTIME_GET_VERSION: {
        struct hmcuda_runtime_get_version_resp resp;
        ctx->cuda_error = hmcuda_core_runtime_get_version(&resp.version);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    case HMCUDA_CMD_FUNC_GET_ATTRIBUTES: {
        struct hmcuda_func_get_attributes_req *req = payload;
        struct hmcuda_func_get_attributes_resp resp;
        ctx->cuda_error = hmcuda_core_func_get_attributes(sid, vid, req->func, &resp.attr);
        memcpy(ctx->resp_payload, &resp, sizeof(resp));
        ctx->resp_payload_len = sizeof(resp);
        break;
    }
    default:
        return -1; /* Not a runtime command */
    }
    return 0;
}
