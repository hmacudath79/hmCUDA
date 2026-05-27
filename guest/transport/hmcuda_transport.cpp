#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <time.h>

#include "hmcuda_api.h"
#include "hmcuda_log.h"
#include "hmcuda_transport.h"

#define DEVICE_PATH "/dev/virtio-hmcuda"
#define HM_CUDA_IOC_MAGIC 'k'
#define HM_CUDA_IOC_REQ _IOWR(HM_CUDA_IOC_MAGIC, 1, struct hmcuda_req_header)

enum TransportType {
    TRANSPORT_VIRTIO,
    TRANSPORT_RPC
};

static const char *hmcuda_cmd_name(uint32_t cmd)
{
    switch (cmd) {
    case HMCUDA_CMD_MALLOC:                          return "MALLOC";
    case HMCUDA_CMD_FREE:                            return "FREE";
    case HMCUDA_CMD_MEMCPY:                          return "MEMCPY";
    case HMCUDA_CMD_MEMSET:                          return "MEMSET";
    case HMCUDA_CMD_LAUNCH_KERNEL:                   return "LAUNCH_KERNEL";
    case HMCUDA_CMD_STREAM_CREATE:                   return "STREAM_CREATE";
    case HMCUDA_CMD_STREAM_SYNCHRONIZE:              return "STREAM_SYNCHRONIZE";
    case HMCUDA_CMD_EVENT_CREATE:                    return "EVENT_CREATE";
    case HMCUDA_CMD_EVENT_RECORD:                    return "EVENT_RECORD";
    case HMCUDA_CMD_EVENT_SYNCHRONIZE:               return "EVENT_SYNCHRONIZE";
    case HMCUDA_CMD_EVENT_ELAPSED_TIME:              return "EVENT_ELAPSED_TIME";
    case HMCUDA_CMD_EVENT_DESTROY:                   return "EVENT_DESTROY";
    case HMCUDA_CMD_DEVICE_SYNCHRONIZE:              return "DEVICE_SYNCHRONIZE";
    case HMCUDA_CMD_INIT:                            return "INIT";
    case HMCUDA_CMD_REGISTER_FATBIN:                 return "REGISTER_FATBIN";
    case HMCUDA_CMD_REGISTER_FUNCTION:               return "REGISTER_FUNCTION";
    case HMCUDA_CMD_PUSH_CALL_CONFIGURATION:         return "PUSH_CALL_CONFIGURATION";
    case HMCUDA_CMD_POP_CALL_CONFIGURATION:          return "POP_CALL_CONFIGURATION";
    case HMCUDA_CMD_INIT_MODULE:                     return "INIT_MODULE";
    case HMCUDA_CMD_GET_DEVICE_PROPERTIES:           return "GET_DEVICE_PROPERTIES";
    case HMCUDA_CMD_GET_DEVICE_COUNT:                return "GET_DEVICE_COUNT";
    case HMCUDA_CMD_SET_DEVICE:                      return "SET_DEVICE";
    case HMCUDA_CMD_GET_ERROR_STRING:                return "GET_ERROR_STRING";
    case HMCUDA_CMD_GET_ERROR_NAME:                  return "GET_ERROR_NAME";
    case HMCUDA_CMD_RUNTIME_GET_VERSION:             return "RUNTIME_GET_VERSION";
    case HMCUDA_CMD_FUNC_GET_ATTRIBUTES:             return "FUNC_GET_ATTRIBUTES";
    case HMCUDA_CMD_CU_INIT:                         return "CU_INIT";
    case HMCUDA_CMD_CU_DRIVER_GET_VERSION:           return "CU_DRIVER_GET_VERSION";
    case HMCUDA_CMD_CU_DEVICE_GET:                   return "CU_DEVICE_GET";
    case HMCUDA_CMD_CU_DEVICE_GET_COUNT:             return "CU_DEVICE_GET_COUNT";
    case HMCUDA_CMD_CU_DEVICE_GET_NAME:              return "CU_DEVICE_GET_NAME";
    case HMCUDA_CMD_CU_DEVICE_GET_ATTRIBUTE:         return "CU_DEVICE_GET_ATTRIBUTE";
    case HMCUDA_CMD_CU_DEVICE_GET_UUID:              return "CU_DEVICE_GET_UUID";
    case HMCUDA_CMD_CU_DEVICE_CAN_ACCESS_PEER:       return "CU_DEVICE_CAN_ACCESS_PEER";
    case HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RETAIN:    return "CU_DEVICE_PRIMARY_CTX_RETAIN";
    case HMCUDA_CMD_CU_DEVICE_PRIMARY_CTX_RELEASE:   return "CU_DEVICE_PRIMARY_CTX_RELEASE";
    case HMCUDA_CMD_CU_CTX_SET_CURRENT:              return "CU_CTX_SET_CURRENT";
    case HMCUDA_CMD_CU_CTX_GET_DEVICE:               return "CU_CTX_GET_DEVICE";
    case HMCUDA_CMD_CU_CTX_ENABLE_PEER_ACCESS:       return "CU_CTX_ENABLE_PEER_ACCESS";
    case HMCUDA_CMD_CU_STREAM_CREATE:                return "CU_STREAM_CREATE";
    case HMCUDA_CMD_CU_STREAM_DESTROY:               return "CU_STREAM_DESTROY";
    case HMCUDA_CMD_CU_STREAM_SYNCHRONIZE:           return "CU_STREAM_SYNCHRONIZE";
    case HMCUDA_CMD_CU_STREAM_QUERY:                 return "CU_STREAM_QUERY";
    case HMCUDA_CMD_CU_STREAM_WAIT_EVENT:            return "CU_STREAM_WAIT_EVENT";
    case HMCUDA_CMD_CU_STREAM_GET_CTX:               return "CU_STREAM_GET_CTX";
    case HMCUDA_CMD_CU_EVENT_CREATE:                 return "CU_EVENT_CREATE";
    case HMCUDA_CMD_CU_EVENT_DESTROY:                return "CU_EVENT_DESTROY";
    case HMCUDA_CMD_CU_EVENT_RECORD:                 return "CU_EVENT_RECORD";
    case HMCUDA_CMD_CU_EVENT_ELAPSED_TIME:           return "CU_EVENT_ELAPSED_TIME";
    case HMCUDA_CMD_CU_MEM_ALLOC:                    return "CU_MEM_ALLOC";
    case HMCUDA_CMD_CU_MEM_FREE:                     return "CU_MEM_FREE";
    case HMCUDA_CMD_CU_MEM_HOST_ALLOC:               return "CU_MEM_HOST_ALLOC";
    case HMCUDA_CMD_CU_MEM_FREE_HOST:                return "CU_MEM_FREE_HOST";
    case HMCUDA_CMD_CU_MEMSET_D32:                   return "CU_MEMSET_D32";
    case HMCUDA_CMD_CU_MEMCPY_ASYNC:                 return "CU_MEMCPY_ASYNC";
    case HMCUDA_CMD_CU_MEMCPY_HTOD:                  return "CU_MEMCPY_HTOD";
    case HMCUDA_CMD_CU_MEMCPY:                       return "CU_MEMCPY";
    case HMCUDA_CMD_CU_POINTER_GET_ATTRIBUTE:        return "CU_POINTER_GET_ATTRIBUTE";
    case HMCUDA_CMD_CU_GET_ERROR_STRING:             return "CU_GET_ERROR_STRING";
    case HMCUDA_CMD_CU_GET_ERROR_NAME:               return "CU_GET_ERROR_NAME";
    case HMCUDA_CMD_CU_MEM_CREATE:                   return "CU_MEM_CREATE";
    case HMCUDA_CMD_CU_MEM_RELEASE:                  return "CU_MEM_RELEASE";
    case HMCUDA_CMD_CU_MEM_MAP:                      return "CU_MEM_MAP";
    case HMCUDA_CMD_CU_MEM_UNMAP:                    return "CU_MEM_UNMAP";
    case HMCUDA_CMD_CU_MEM_ADDRESS_RESERVE:          return "CU_MEM_ADDRESS_RESERVE";
    case HMCUDA_CMD_CU_MEM_ADDRESS_FREE:             return "CU_MEM_ADDRESS_FREE";
    case HMCUDA_CMD_CU_MEM_EXPORT_TO_SHAREABLE_HANDLE:   return "CU_MEM_EXPORT_TO_SHAREABLE_HANDLE";
    case HMCUDA_CMD_CU_MEM_IMPORT_FROM_SHAREABLE_HANDLE: return "CU_MEM_IMPORT_FROM_SHAREABLE_HANDLE";
    case HMCUDA_CMD_CU_MEM_SET_ACCESS:               return "CU_MEM_SET_ACCESS";
    case HMCUDA_CMD_CU_MEM_GET_ALLOC_GRANULARITY:    return "CU_MEM_GET_ALLOC_GRANULARITY";
    case HMCUDA_CMD_CU_MULTICAST_CREATE:             return "CU_MULTICAST_CREATE";
    case HMCUDA_CMD_CU_MULTICAST_GET_GRANULARITY:    return "CU_MULTICAST_GET_GRANULARITY";
    case HMCUDA_CMD_CU_MULTICAST_ADD_DEVICE:         return "CU_MULTICAST_ADD_DEVICE";
    case HMCUDA_CMD_CU_MULTICAST_BIND_MEM:           return "CU_MULTICAST_BIND_MEM";
    case HMCUDA_CMD_CU_MULTICAST_UNBIND:             return "CU_MULTICAST_UNBIND";
    default: {
        static __thread char buf[16];
        snprintf(buf, sizeof(buf), "0x%x", cmd);
        return buf;
    }
    }
}

static TransportType g_transport = TRANSPORT_VIRTIO;
static int g_fd = -1;

/* Global log level - initialized from HMCUDA_DEBUG env var */
HmcudaLogLevel g_hmcuda_log_level = HMCUDA_LOG_INFO;

void hmcuda_transport_init()
{
    /* Read log level from HMCUDA_DEBUG:
     * 0=NONE 1=ERROR 2=WARN 3=INFO 4=DEBUG 5=TIMING only */
    const char *debug_env = getenv("HMCUDA_DEBUG");
    if (debug_env) {
        int level = atoi(debug_env);
        if (level >= HMCUDA_LOG_NONE && level <= HMCUDA_LOG_TIMING)
            g_hmcuda_log_level = (HmcudaLogLevel)level;
    }

    const char *env = getenv("HMCUDA_TRANSPORT");
    if (env && strcmp(env, "rpc") == 0) {
        g_transport = TRANSPORT_RPC;
        HMCUDA_LOG_INFO("Transport initialized (RPC)");
    } else {
        if (g_fd < 0) {
            g_fd = open(DEVICE_PATH, O_RDWR);
            if (g_fd < 0) {
                perror("hmCUDA: Failed to open device");
                return;
            }
            HMCUDA_LOG_INFO("Transport initialized (VirtIO)");
        }
    }
}

void hmcuda_transport_fini()
{
    if (g_fd >= 0) {
        close(g_fd);
        g_fd = -1;
    }
}

bool hmcuda_transport_ready()
{
    return g_fd >= 0 || g_transport == TRANSPORT_RPC;
}

uint32_t hmcuda_transport_dispatch(uint32_t cmd,
                                    const void *req, size_t req_len,
                                    void *resp, size_t resp_len)
{
    if (g_transport == TRANSPORT_RPC) {
        HMCUDA_LOG_ERROR("hmCUDA: RPC transport not implemented yet.");
        return 1; /* cudaErrorUnknown / CUDA_ERROR_UNKNOWN */
    }

    if (g_fd < 0) return 3; /* cudaErrorInitializationError */

    size_t payload_len = (req_len > resp_len) ? req_len : resp_len;
    size_t total_len = sizeof(hmcuda_req_header) + payload_len;

    if (total_len < sizeof(hmcuda_resp_header) + resp_len)
        total_len = sizeof(hmcuda_resp_header) + resp_len;

    uint8_t stack_buf[1024];
    uint8_t *buffer_ptr = stack_buf;
    std::vector<uint8_t> heap_buf;

    if (total_len > sizeof(stack_buf)) {
        heap_buf.resize(total_len);
        buffer_ptr = heap_buf.data();
    }

    hmcuda_req_header *hdr = (hmcuda_req_header *)buffer_ptr;
    hdr->cmd_type = cmd;
    hdr->flags = 0;

    if (req && req_len > 0)
        memcpy(buffer_ptr + sizeof(hmcuda_req_header), req, req_len);

    struct timespec t0, t1;
    if (HMCUDA_LOG_ENABLED(HMCUDA_LOG_TIMING))
        clock_gettime(CLOCK_MONOTONIC, &t0);

    if (ioctl(g_fd, HM_CUDA_IOC_REQ, buffer_ptr) < 0) {
        perror("hmCUDA: ioctl failed");
        return 1;
    }

    if (HMCUDA_LOG_ENABLED(HMCUDA_LOG_TIMING)) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long long elapsed_us = (t1.tv_sec - t0.tv_sec) * 1000000LL
                             + (t1.tv_nsec - t0.tv_nsec) / 1000;
        HMCUDA_LOG_TIMING(hmcuda_cmd_name(cmd), elapsed_us);
    }

    hmcuda_resp_header *resp_hdr = (hmcuda_resp_header *)buffer_ptr;
    if (resp_hdr->cuda_error != 0)
        return resp_hdr->cuda_error;

    if (resp && resp_len > 0)
        memcpy(resp, buffer_ptr + sizeof(hmcuda_resp_header), resp_len);

    return 0;
}

/* =========================================================================
 * Host-memory registry (cuMemHostAlloc buffers)
 *
 * Both libcuda.so and libcudart.so link against hmcuda_transport, so this
 * single map is shared between them without any cross-library coupling.
 * ========================================================================= */

struct HostMemEntry {
    size_t size;
    bool   gpu_dirty; /* true if a GPU kernel may have written to real_ptr since last H2D sync */
};
static std::unordered_map<const void *, HostMemEntry> g_host_mem_map;

void hmcuda_transport_host_mem_register(const void *guest_va, size_t size)
{
    g_host_mem_map[guest_va] = { size, false };
}

void hmcuda_transport_host_mem_unregister(const void *guest_va)
{
    g_host_mem_map.erase(guest_va);
}

size_t hmcuda_transport_host_mem_size(const void *guest_va)
{
    auto it = g_host_mem_map.find(guest_va);
    return (it != g_host_mem_map.end()) ? it->second.size : 0;
}

void hmcuda_transport_mark_gpu_dirty(const void *guest_va)
{
    auto it = g_host_mem_map.find(guest_va);
    if (it != g_host_mem_map.end())
        it->second.gpu_dirty = true;
}

bool hmcuda_transport_is_gpu_dirty(const void *guest_va)
{
    auto it = g_host_mem_map.find(guest_va);
    return (it != g_host_mem_map.end()) && it->second.gpu_dirty;
}

void hmcuda_transport_clear_gpu_dirty(const void *guest_va)
{
    auto it = g_host_mem_map.find(guest_va);
    if (it != g_host_mem_map.end())
        it->second.gpu_dirty = false;
}

void hmcuda_transport_sync_host_mem_arg(const void *ptr)
{
    auto it = g_host_mem_map.find(ptr);
    if (it == g_host_mem_map.end()) return;

    /* If a GPU kernel wrote to real_ptr after the last sync, real_ptr is
     * authoritative — do not overwrite it with stale guest_va content. */
    if (it->second.gpu_dirty) return;

    size_t alloc_size = it->second.size;
    HMCUDA_LOG_INFO("hmcuda_transport_sync_host_mem_arg: syncing guest_va=%p size=%zu",
                    ptr, alloc_size);

    struct hmcuda_memcpy_req req;
    req.dst   = (uint64_t)(uintptr_t)ptr;
    req.src   = (uint64_t)(uintptr_t)ptr;
    req.count = (uint64_t)alloc_size;
    req.kind  = 1; /* H2D */
    hmcuda_transport_dispatch(HMCUDA_CMD_CU_MEMCPY, &req, sizeof(req), nullptr, 0);
}
