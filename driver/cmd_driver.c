// SPDX-License-Identifier: GPL-2.0-only
#include <linux/types.h>
#include <linux/errno.h>

#include "hmcuda_api.h"
#include "cmd_dispatch.h"

/*
 * CUDA Driver API — command size lookup.
 *
 * Same two-pass protocol as cmd_runtime.c:
 *   1. req_payload == NULL → return fixed struct sizes.
 *   2. req_payload != NULL → add variable-length trailing data.
 */
int hmcuda_driver_get_sizes(uint32_t cmd, const void *req_payload,
                            struct hmcuda_cmd_sizes *sizes)
{
    sizes->req_size = 0;
    sizes->resp_size = 0;

    switch (cmd) {
    /* --- Initialization & version --- */
    case HMCUDA_CMD_CU_INIT:
        sizes->req_size = sizeof(struct hmcuda_cu_init_req);
        break;
    case HMCUDA_CMD_CU_DRIVER_GET_VERSION:
        sizes->resp_size = sizeof(struct hmcuda_cu_driver_get_version_resp);
        break;

    /* --- Device management --- */
    case HMCUDA_CMD_CU_DEVICE_GET:
        sizes->req_size = sizeof(struct hmcuda_cu_device_get_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_device_get_resp);
        break;
    case HMCUDA_CMD_CU_DEVICE_GET_COUNT:
        sizes->resp_size = sizeof(struct hmcuda_cu_device_get_count_resp);
        break;
    case HMCUDA_CMD_CU_DEVICE_GET_NAME:
        sizes->req_size = sizeof(struct hmcuda_cu_device_get_name_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_device_get_name_resp);
        if (req_payload) {
            const struct hmcuda_cu_device_get_name_req *r = req_payload;
            sizes->resp_size += r->name_len;
        }
        break;
    case HMCUDA_CMD_CU_DEVICE_GET_ATTRIBUTE:
        sizes->req_size = sizeof(struct hmcuda_cu_device_get_attribute_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_device_get_attribute_resp);
        break;
    case HMCUDA_CMD_CU_DEVICE_GET_UUID:
        sizes->req_size = sizeof(struct hmcuda_cu_device_get_uuid_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_device_get_uuid_resp);
        break;
    case HMCUDA_CMD_CU_DEVICE_CAN_ACCESS_PEER:
        sizes->req_size = sizeof(struct hmcuda_cu_device_can_access_peer_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_device_can_access_peer_resp);
        break;
    case HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RETAIN:
        sizes->req_size = sizeof(struct hmcuda_cu_device_primary_ctx_retain_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_device_primary_ctx_retain_resp);
        break;
    case HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RELEASE:
        sizes->req_size = sizeof(struct hmcuda_cu_device_primary_ctx_release_req);
        break;

    /* --- Context management --- */
    case HMCUDA_CMD_CU_CTX_SET_CURRENT:
        sizes->req_size = sizeof(struct hmcuda_cu_ctx_set_current_req);
        break;
    case HMCUDA_CMD_CU_CTX_GET_DEVICE:
        sizes->resp_size = sizeof(struct hmcuda_cu_ctx_get_device_resp);
        break;
    case HMCUDA_CMD_CU_CTX_ENABLE_PEER_ACCESS:
        sizes->req_size = sizeof(struct hmcuda_cu_ctx_enable_peer_access_req);
        break;

    /* --- Stream management --- */
    case HMCUDA_CMD_CU_STREAM_CREATE:
        sizes->req_size = sizeof(struct hmcuda_cu_stream_create_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_stream_create_resp);
        break;
    case HMCUDA_CMD_CU_STREAM_DESTROY:
        sizes->req_size = sizeof(struct hmcuda_cu_stream_destroy_req);
        break;
    case HMCUDA_CMD_CU_STREAM_SYNCHRONIZE:
        sizes->req_size = sizeof(struct hmcuda_cu_stream_synchronize_req);
        break;
    case HMCUDA_CMD_CU_STREAM_QUERY:
        sizes->req_size = sizeof(struct hmcuda_cu_stream_query_req);
        break;
    case HMCUDA_CMD_CU_STREAM_WAIT_EVENT:
        sizes->req_size = sizeof(struct hmcuda_cu_stream_wait_event_req);
        break;
    case HMCUDA_CMD_CU_STREAM_GET_CTX:
        sizes->req_size = sizeof(struct hmcuda_cu_stream_get_ctx_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_stream_get_ctx_resp);
        break;

    /* --- Event management --- */
    case HMCUDA_CMD_CU_EVENT_CREATE:
        sizes->req_size = sizeof(struct hmcuda_cu_event_create_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_event_create_resp);
        break;
    case HMCUDA_CMD_CU_EVENT_DESTROY:
        sizes->req_size = sizeof(struct hmcuda_cu_event_destroy_req);
        break;
    case HMCUDA_CMD_CU_EVENT_RECORD:
        sizes->req_size = sizeof(struct hmcuda_cu_event_record_req);
        break;
    case HMCUDA_CMD_CU_EVENT_ELAPSED_TIME:
        sizes->req_size = sizeof(struct hmcuda_cu_event_elapsed_time_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_event_elapsed_time_resp);
        break;

    /* --- Memory allocation --- */
    case HMCUDA_CMD_CU_MEM_ALLOC:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_alloc_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_mem_alloc_resp);
        break;
    case HMCUDA_CMD_CU_MEM_FREE:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_free_req);
        break;
    case HMCUDA_CMD_CU_MEM_HOST_ALLOC:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_host_alloc_req);
        break;
    case HMCUDA_CMD_CU_MEM_FREE_HOST:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_free_host_req);
        break;
    case HMCUDA_CMD_CU_MEMSET_D32:
        sizes->req_size = sizeof(struct hmcuda_cu_memset_d32_req);
        break;

    /* --- Memory copy --- */
    case HMCUDA_CMD_CU_MEMCPY_ASYNC:
        sizes->req_size = sizeof(struct hmcuda_cu_memcpy_async_req);
        break;
    case HMCUDA_CMD_CU_MEMCPY_HTOD:
        sizes->req_size = sizeof(struct hmcuda_cu_memcpy_htod_req);
        break;
    case HMCUDA_CMD_CU_MEMCPY:
        sizes->req_size = sizeof(struct hmcuda_memcpy_req);
        break;

    /* --- Pointer & error --- */
    case HMCUDA_CMD_CU_POINTER_GET_ATTRIBUTE:
        sizes->req_size = sizeof(struct hmcuda_cu_pointer_get_attribute_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_pointer_get_attribute_resp);
        break;
    case HMCUDA_CMD_CU_GET_ERROR_STRING:
        sizes->req_size = sizeof(struct hmcuda_cu_get_error_string_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_get_error_string_resp);
        /* Variable-length string appended by host; guest allocates max */
        break;
    case HMCUDA_CMD_CU_GET_ERROR_NAME:
        sizes->req_size = sizeof(struct hmcuda_cu_get_error_name_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_get_error_name_resp);
        break;

    /* --- Virtual memory management --- */
    case HMCUDA_CMD_CU_MEM_CREATE:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_create_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_mem_create_resp);
        break;
    case HMCUDA_CMD_CU_MEM_RELEASE:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_release_req);
        break;
    case HMCUDA_CMD_CU_MEM_MAP:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_map_req);
        break;
    case HMCUDA_CMD_CU_MEM_UNMAP:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_unmap_req);
        break;
    case HMCUDA_CMD_CU_MEM_ADDRESS_RESERVE:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_address_reserve_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_mem_address_reserve_resp);
        break;
    case HMCUDA_CMD_CU_MEM_ADDRESS_FREE:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_address_free_req);
        break;
    case HMCUDA_CMD_CU_MEM_SET_ACCESS:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_set_access_req);
        if (req_payload) {
            const struct hmcuda_cu_mem_set_access_req *r = req_payload;
            sizes->req_size += r->count * sizeof(struct hmcuda_cu_mem_access_desc);
        }
        break;
    case HMCUDA_CMD_CU_MEM_GET_ALLOC_GRANULARITY:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_get_alloc_granularity_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_mem_get_alloc_granularity_resp);
        break;
    case HMCUDA_CMD_CU_MEM_EXPORT_TO_SHAREABLE_HANDLE:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_export_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_mem_export_resp);
        break;
    case HMCUDA_CMD_CU_MEM_IMPORT_FROM_SHAREABLE_HANDLE:
        sizes->req_size = sizeof(struct hmcuda_cu_mem_import_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_mem_import_resp);
        break;

    /* --- Multicast memory --- */
    case HMCUDA_CMD_CU_MULTICAST_CREATE:
        sizes->req_size = sizeof(struct hmcuda_cu_multicast_create_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_multicast_create_resp);
        break;
    case HMCUDA_CMD_CU_MULTICAST_GET_GRANULARITY:
        sizes->req_size = sizeof(struct hmcuda_cu_multicast_get_granularity_req);
        sizes->resp_size = sizeof(struct hmcuda_cu_multicast_get_granularity_resp);
        break;
    case HMCUDA_CMD_CU_MULTICAST_ADD_DEVICE:
        sizes->req_size = sizeof(struct hmcuda_cu_multicast_add_device_req);
        break;
    case HMCUDA_CMD_CU_MULTICAST_BIND_MEM:
        sizes->req_size = sizeof(struct hmcuda_cu_multicast_bind_mem_req);
        break;
    case HMCUDA_CMD_CU_MULTICAST_UNBIND:
        sizes->req_size = sizeof(struct hmcuda_cu_multicast_unbind_req);
        break;

    default:
        return -EINVAL;
    }
    return 0;
}
