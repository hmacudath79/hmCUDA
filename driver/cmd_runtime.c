// SPDX-License-Identifier: GPL-2.0-only
#include <linux/types.h>
#include <linux/errno.h>

#include "hmcuda_api.h"
#include "cmd_dispatch.h"

/*
 * CUDA Runtime API — command size lookup.
 *
 * Called by the main ioctl handler in virtio_hmcuda_main.c to determine
 * how much data to copy from userspace before submitting to the virtqueue.
 *
 * This function is called twice per ioctl:
 *   1. First with req_payload == NULL to get the fixed struct sizes.
 *   2. Then with req_payload pointing to the fixed struct (already copied
 *      from userspace) so we can add any variable-length trailing data.
 */
int hmcuda_runtime_get_sizes(uint32_t cmd, const void *req_payload,
                             struct hmcuda_cmd_sizes *sizes)
{
    sizes->req_size = 0;
    sizes->resp_size = 0;

    switch (cmd) {
    case HMCUDA_CMD_MALLOC:
        sizes->req_size = sizeof(struct hmcuda_malloc_req);
        sizes->resp_size = sizeof(struct hmcuda_malloc_resp);
        break;
    case HMCUDA_CMD_FREE:
        sizes->req_size = sizeof(struct hmcuda_free_req);
        break;
    case HMCUDA_CMD_MEMCPY:
        sizes->req_size = sizeof(struct hmcuda_memcpy_req);
        break;
    case HMCUDA_CMD_MEMSET:
        sizes->req_size = sizeof(struct hmcuda_memset_req);
        break;
    case HMCUDA_CMD_LAUNCH_KERNEL:
        sizes->req_size = sizeof(struct hmcuda_launch_kernel_req);
        if (req_payload) {
            const struct hmcuda_launch_kernel_req *r = req_payload;
            sizes->req_size += r->args_size;
        }
        break;
    case HMCUDA_CMD_STREAM_CREATE:
        sizes->req_size = sizeof(struct hmcuda_stream_create_req);
        sizes->resp_size = sizeof(struct hmcuda_stream_create_resp);
        break;
    case HMCUDA_CMD_STREAM_SYNCHRONIZE:
        sizes->req_size = sizeof(struct hmcuda_stream_synchronize_req);
        break;
    case HMCUDA_CMD_EVENT_CREATE:
        sizes->req_size = sizeof(struct hmcuda_event_create_req);
        sizes->resp_size = sizeof(struct hmcuda_event_create_resp);
        break;
    case HMCUDA_CMD_EVENT_RECORD:
        sizes->req_size = sizeof(struct hmcuda_event_record_req);
        break;
    case HMCUDA_CMD_EVENT_SYNCHRONIZE:
        sizes->req_size = sizeof(struct hmcuda_event_synchronize_req);
        break;
    case HMCUDA_CMD_EVENT_ELAPSED_TIME:
        sizes->req_size = sizeof(struct hmcuda_event_elapsed_time_req);
        sizes->resp_size = sizeof(struct hmcuda_event_elapsed_time_resp);
        break;
    case HMCUDA_CMD_EVENT_DESTROY:
        sizes->req_size = sizeof(struct hmcuda_event_destroy_req);
        break;
    case HMCUDA_CMD_DEVICE_SYNCHRONIZE:
        sizes->req_size = sizeof(struct hmcuda_device_synchronize_req);
        break;
    case HMCUDA_CMD_INIT:
        break;
    case HMCUDA_CMD_PUSH_CALL_CONFIGURATION:
        sizes->req_size = sizeof(struct hmcuda_push_call_cfg_req);
        break;
    case HMCUDA_CMD_POP_CALL_CONFIGURATION:
        sizes->resp_size = sizeof(struct hmcuda_pop_call_cfg_resp);
        break;
    case HMCUDA_CMD_INIT_MODULE:
        sizes->req_size = sizeof(struct hmcuda_init_module_req);
        break;
    case HMCUDA_CMD_GET_DEVICE_PROPERTIES:
        sizes->req_size = sizeof(struct hmcuda_get_device_properties_req);
        sizes->resp_size = sizeof(struct hmcuda_get_device_properties_resp);
        break;
    case HMCUDA_CMD_GET_DEVICE_COUNT:
        sizes->resp_size = sizeof(struct hmcuda_get_device_count_resp);
        break;
    case HMCUDA_CMD_SET_DEVICE:
        sizes->req_size = sizeof(struct hmcuda_set_device_req);
        break;
    case HMCUDA_CMD_GET_ERROR_STRING:
    case HMCUDA_CMD_GET_ERROR_NAME:
        sizes->req_size = sizeof(struct hmcuda_get_error_string_req);
        sizes->resp_size = sizeof(struct hmcuda_get_error_string_resp) + 256;
        break;
    case HMCUDA_CMD_REGISTER_FATBIN:
        sizes->req_size = sizeof(struct hmcuda_register_fatbin_req);
        sizes->resp_size = sizeof(struct hmcuda_register_fatbin_resp);
        if (req_payload) {
            const struct hmcuda_register_fatbin_req *r = req_payload;
            sizes->req_size += r->size;
        }
        break;
    case HMCUDA_CMD_REGISTER_FUNCTION:
        sizes->req_size = sizeof(struct hmcuda_register_function_req);
        sizes->resp_size = sizeof(struct hmcuda_register_function_resp);
        if (req_payload) {
            const struct hmcuda_register_function_req *r = req_payload;
            sizes->req_size += r->device_name_len;
        }
        break;
    case HMCUDA_CMD_RUNTIME_GET_VERSION:
        sizes->resp_size = sizeof(struct hmcuda_runtime_get_version_resp);
        break;
    case HMCUDA_CMD_FUNC_GET_ATTRIBUTES:
        sizes->req_size = sizeof(struct hmcuda_func_get_attributes_req);
        sizes->resp_size = sizeof(struct hmcuda_func_get_attributes_resp);
        break;
    default:
        return -EINVAL;
    }
    return 0;
}
