#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cxxabi.h>
#include <cuda_runtime.h>

#include "hmcuda_api.h"
#include "hmcuda_log.h"
#include "hmcuda_transport.h"
#include "hmcuda_dispatch.h"

// Function info: handle + param count for arg marshalling
struct FuncInfo {
    uint64_t handle;
    uint32_t param_count;
    uint32_t param_total_size;
};
static std::unordered_map<const void*, FuncInfo> g_func_map;

static thread_local cudaError_t g_last_error = cudaSuccess;

static cudaError_t hmcuda_dispatch(uint32_t cmd, const void* req, size_t req_len, void* resp, size_t resp_len) {
    cudaError_t err = (cudaError_t)hmcuda_transport_dispatch(cmd, req, req_len, resp, resp_len);
    g_last_error = err;
    return err;
}

/* Short aliases — bake in the dispatch function for this file */
#define RT_FWD(cmd, req)       HMCUDA_FORWARD(hmcuda_dispatch, cmd, req)
#define RT_FWD_NOREQ(cmd)     HMCUDA_FORWARD_NOREQ(hmcuda_dispatch, cmd)

#define RT_FWD_RESP(cmd, req, resp_t, out, cast, field) \
    HMCUDA_FORWARD_RESP(hmcuda_dispatch, cmd, req, resp_t, out, cast, field)

#define RT_FWD_NOREQ_RESP(cmd, resp_t, out, cast, field) \
    HMCUDA_FORWARD_NOREQ_RESP(hmcuda_dispatch, cmd, resp_t, out, cast, field)

// Count kernel parameters by demangling the device function name.
// e.g. "_Z6vecAddPKfS0_Pfi" → "vecAdd(float const*, float const*, float*, int)" → 4 params
static uint32_t count_params_from_mangled(const char *mangled) {
    int status = 0;
    char *demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status != 0 || !demangled) {
        HMCUDA_LOG_ERROR("Warning: could not demangle '%s'", mangled);
        free(demangled);
        return 0;
    }

    const char *paren = strchr(demangled, '(');
    if (!paren) {
        free(demangled);
        return 0;
    }
    paren++;

    const char *end = strrchr(demangled, ')');
    if (!end || end <= paren) {
        free(demangled);
        return 0;
    }

    // Count params by counting commas at depth 0 (skip nested <> and ())
    uint32_t count = 1;
    int depth = 0;
    for (const char *p = paren; p < end; p++) {
        if (*p == '<' || *p == '(') depth++;
        else if (*p == '>' || *p == ')') depth--;
        else if (*p == ',' && depth == 0) count++;
    }

    if (count == 1) {
        while (*paren == ' ') paren++;
        if (strncmp(paren, "void", 4) == 0) count = 0;
    }

    HMCUDA_LOG_INFO("Demangled '%s' → '%s' → %u params", mangled, demangled, count);
    free(demangled);
    return count;
}

static void __attribute__((constructor(101))) hmcuda_runtime_init() {
    hmcuda_transport_init();

    if (hmcuda_dispatch(HMCUDA_CMD_INIT, nullptr, 0, nullptr, 0) != cudaSuccess) {
        HMCUDA_LOG_WARN("Failed to initialize host runtime");
    }
}

static void __attribute__((destructor)) hmcuda_runtime_fini() {
    hmcuda_transport_fini();
}

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    HMCUDA_LOG_INFO("cudaMalloc(size=%zu)", size);
    hmcuda_malloc_req req = { .size = size };
    hmcuda_malloc_resp resp;
    cudaError_t err = hmcuda_dispatch(HMCUDA_CMD_MALLOC, &req, sizeof(req), &resp, sizeof(resp));
    if (err == cudaSuccess && devPtr) {
        *devPtr = (void*)resp.devPtr;
    }
    return err;
}

cudaError_t cudaFree(void *devPtr) {
    HMCUDA_LOG_INFO("cudaFree(ptr=%p)", devPtr);
    hmcuda_free_req req = { .devPtr = (uint64_t)devPtr };
    RT_FWD(HMCUDA_CMD_FREE, req);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    HMCUDA_LOG_INFO("cudaMemcpy(dst=%p, src=%p, count=%zu, kind=%d)", dst, src, count, kind);
    hmcuda_memcpy_req req = {
        .dst = (uint64_t)dst,
        .src = (uint64_t)src,
        .count = count,
        .kind = (uint32_t)kind
    };
    RT_FWD(HMCUDA_CMD_MEMCPY, req);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    HMCUDA_LOG_INFO("cudaMemset(ptr=%p, val=%d, count=%zu)", devPtr, value, count);
    hmcuda_memset_req req = {
        .devPtr = (uint64_t)devPtr,
        .value = value,
        .count = count
    };
    RT_FWD(HMCUDA_CMD_MEMSET, req);
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
    HMCUDA_LOG_INFO("cudaLaunchKernel(func=%p)", func);

    auto it = g_func_map.find(func);
    if (it == g_func_map.end()) {
        HMCUDA_LOG_ERROR("cudaLaunchKernel: Unknown function pointer %p", func);
        g_last_error = cudaErrorInvalidDeviceFunction;
        return cudaErrorInvalidDeviceFunction;
    }

    const FuncInfo &fi = it->second;
    uint32_t param_count = fi.param_count;

    uint32_t args_size = param_count * 8;
    std::vector<uint8_t> payload(sizeof(hmcuda_launch_kernel_req) + args_size, 0);

    hmcuda_launch_kernel_req *req = (hmcuda_launch_kernel_req *)payload.data();
    req->func = fi.handle;
    req->gridDim = { gridDim.x, gridDim.y, gridDim.z };
    req->blockDim = { blockDim.x, blockDim.y, blockDim.z };
    req->sharedMem = (uint32_t)sharedMem;
    req->stream = (uint64_t)stream;
    req->args_size = args_size;

    uint8_t *arg_buf = payload.data() + sizeof(hmcuda_launch_kernel_req);
    if (args) {
        for (uint32_t i = 0; i < param_count; i++) {
            memcpy(arg_buf + i * 8, args[i], 8);
        }
    }

    HMCUDA_LOG_INFO("cudaLaunchKernel: handle=0x%lx, param_count=%u, args_size=%u",
           fi.handle, param_count, args_size);

    /* Before launching, sync any cuMemHostAlloc buffers passed as args:
     * push CPU-written guest_va → real_ptr and clear gpu_dirty. */
    if (args) {
        for (uint32_t i = 0; i < param_count; i++) {
            uint64_t v = 0;
            memcpy(&v, args[i], 8);
            hmcuda_transport_sync_host_mem_arg((const void *)(uintptr_t)v);
        }
    }

    cudaError_t err = hmcuda_dispatch(HMCUDA_CMD_LAUNCH_KERNEL, payload.data(), payload.size(), nullptr, 0);

    /* After dispatch, mark each HOST_MEM arg as GPU-dirty: the kernel may
     * write to real_ptr, making it diverge from guest_va. */
    if (args) {
        for (uint32_t i = 0; i < param_count; i++) {
            uint64_t v = 0;
            memcpy(&v, args[i], 8);
            hmcuda_transport_mark_gpu_dirty((const void *)(uintptr_t)v);
        }
    }

    return err;
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    HMCUDA_LOG_INFO("cudaStreamCreate");
    hmcuda_stream_create_req req = { .flags = 0 };
    RT_FWD_RESP(HMCUDA_CMD_STREAM_CREATE, req,
        hmcuda_stream_create_resp, pStream, (cudaStream_t), stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    HMCUDA_LOG_INFO("cudaStreamSynchronize(stream=%p)", stream);
    hmcuda_stream_synchronize_req req = { .stream = (uint64_t)stream };
    RT_FWD(HMCUDA_CMD_STREAM_SYNCHRONIZE, req);
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    HMCUDA_LOG_INFO("cudaEventCreate");
    hmcuda_event_create_req req = { .flags = 0 };
    RT_FWD_RESP(HMCUDA_CMD_EVENT_CREATE, req,
        hmcuda_event_create_resp, event, (cudaEvent_t), event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    HMCUDA_LOG_INFO("cudaEventRecord(event=%p, stream=%p)", event, stream);
    hmcuda_event_record_req req = {
        .event = (uint64_t)event,
        .stream = (uint64_t)stream
    };
    RT_FWD(HMCUDA_CMD_EVENT_RECORD, req);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    HMCUDA_LOG_INFO("cudaEventSynchronize(event=%p)", event);
    hmcuda_event_synchronize_req req = { .event = (uint64_t)event };
    RT_FWD(HMCUDA_CMD_EVENT_SYNCHRONIZE, req);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    HMCUDA_LOG_INFO("cudaEventElapsedTime(start=%p, end=%p)", start, end);
    hmcuda_event_elapsed_time_req req = {
        .start_event = (uint64_t)start,
        .end_event = (uint64_t)end
    };
    RT_FWD_RESP(HMCUDA_CMD_EVENT_ELAPSED_TIME, req,
        hmcuda_event_elapsed_time_resp, ms, (float), ms);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    HMCUDA_LOG_INFO("cudaEventDestroy(event=%p)", event);
    hmcuda_event_destroy_req req = { .event = (uint64_t)event };
    RT_FWD(HMCUDA_CMD_EVENT_DESTROY, req);
}

cudaError_t cudaSetDevice(int device) {
    HMCUDA_LOG_INFO("cudaSetDevice(device=%d)", device);
    hmcuda_set_device_req req = { .device = device };
    RT_FWD(HMCUDA_CMD_SET_DEVICE, req);
}

cudaError_t cudaDeviceSynchronize() {
    HMCUDA_LOG_INFO("cudaDeviceSynchronize");
    hmcuda_device_synchronize_req req = { .flags = 0 };
    RT_FWD(HMCUDA_CMD_DEVICE_SYNCHRONIZE, req);
}

const char* cudaGetErrorString(cudaError_t error) {
    HMCUDA_LOG_DEBUG("cudaGetErrorString(error=%d)", (int)error);
    hmcuda_get_error_string_req req = { .error = (uint32_t)error };

    static thread_local char tl_buf[256];
    uint8_t resp_buf[sizeof(hmcuda_get_error_string_resp) + 256];
    cudaError_t err = hmcuda_dispatch(HMCUDA_CMD_GET_ERROR_STRING,
                                       &req, sizeof(req), resp_buf, sizeof(resp_buf));
    if (err == cudaSuccess) {
        hmcuda_get_error_string_resp *resp = (hmcuda_get_error_string_resp *)resp_buf;
        if (resp->str_len > 0) {
            size_t copy = resp->str_len < sizeof(tl_buf) ? resp->str_len : sizeof(tl_buf) - 1;
            memcpy(tl_buf, resp_buf + sizeof(*resp), copy);
            tl_buf[copy] = '\0';
            return tl_buf;
        }
    }
    return "Unknown CUDA error";
}

const char* cudaGetErrorName(cudaError_t error) {
    HMCUDA_LOG_DEBUG("cudaGetErrorName(error=%d)", (int)error);
    hmcuda_get_error_string_req req = { .error = (uint32_t)error };

    static thread_local char tl_buf[256];
    uint8_t resp_buf[sizeof(hmcuda_get_error_string_resp) + 256];
    cudaError_t err = hmcuda_dispatch(HMCUDA_CMD_GET_ERROR_NAME,
                                       &req, sizeof(req), resp_buf, sizeof(resp_buf));
    if (err == cudaSuccess) {
        hmcuda_get_error_string_resp *resp = (hmcuda_get_error_string_resp *)resp_buf;
        if (resp->str_len > 0) {
            size_t copy = resp->str_len < sizeof(tl_buf) ? resp->str_len : sizeof(tl_buf) - 1;
            memcpy(tl_buf, resp_buf + sizeof(*resp), copy);
            tl_buf[copy] = '\0';
            return tl_buf;
        }
    }
    return "UNKNOWN";
}

cudaError_t cudaGetDeviceCount(int *count) {
    HMCUDA_LOG_INFO("cudaGetDeviceCount");
    RT_FWD_NOREQ_RESP(HMCUDA_CMD_GET_DEVICE_COUNT,
        hmcuda_get_device_count_resp, count, (int), count);
}

static void hmcuda_prop_to_cuda(const hmcuda_device_prop *src, cudaDeviceProp *dst) {
    memset(dst, 0, sizeof(*dst));
    memcpy(dst->name, src->name, sizeof(dst->name));
    memcpy(&dst->uuid, src->uuid, 16);
    memcpy(dst->luid, src->luid, 8);
    dst->luidDeviceNodeMask = src->luidDeviceNodeMask;
    dst->totalGlobalMem = src->totalGlobalMem;
    dst->sharedMemPerBlock = src->sharedMemPerBlock;
    dst->regsPerBlock = src->regsPerBlock;
    dst->warpSize = src->warpSize;
    dst->memPitch = src->memPitch;
    dst->maxThreadsPerBlock = src->maxThreadsPerBlock;
    memcpy(dst->maxThreadsDim, src->maxThreadsDim, sizeof(dst->maxThreadsDim));
    memcpy(dst->maxGridSize, src->maxGridSize, sizeof(dst->maxGridSize));
#if CUDART_VERSION < 13000
    dst->clockRate = src->clockRate;
#endif
    dst->totalConstMem = src->totalConstMem;
    dst->major = src->major;
    dst->minor = src->minor;
    dst->textureAlignment = src->textureAlignment;
    dst->texturePitchAlignment = src->texturePitchAlignment;
#if CUDART_VERSION < 13000
    dst->deviceOverlap = src->deviceOverlap;
#endif
    dst->multiProcessorCount = src->multiProcessorCount;
#if CUDART_VERSION < 13000
    dst->kernelExecTimeoutEnabled = src->kernelExecTimeoutEnabled;
#endif
    dst->integrated = src->integrated;
    dst->canMapHostMemory = src->canMapHostMemory;
#if CUDART_VERSION < 13000
    dst->computeMode = src->computeMode;
#endif
    dst->maxTexture1D = src->maxTexture1D;
    dst->maxTexture1DMipmap = src->maxTexture1DMipmap;
#if CUDART_VERSION < 13000
    dst->maxTexture1DLinear = src->maxTexture1DLinear;
#endif
    memcpy(dst->maxTexture2D, src->maxTexture2D, sizeof(dst->maxTexture2D));
    memcpy(dst->maxTexture2DMipmap, src->maxTexture2DMipmap, sizeof(dst->maxTexture2DMipmap));
    memcpy(dst->maxTexture2DLinear, src->maxTexture2DLinear, sizeof(dst->maxTexture2DLinear));
    memcpy(dst->maxTexture2DGather, src->maxTexture2DGather, sizeof(dst->maxTexture2DGather));
    memcpy(dst->maxTexture3D, src->maxTexture3D, sizeof(dst->maxTexture3D));
    memcpy(dst->maxTexture3DAlt, src->maxTexture3DAlt, sizeof(dst->maxTexture3DAlt));
    dst->maxTextureCubemap = src->maxTextureCubemap;
    memcpy(dst->maxTexture1DLayered, src->maxTexture1DLayered, sizeof(dst->maxTexture1DLayered));
    memcpy(dst->maxTexture2DLayered, src->maxTexture2DLayered, sizeof(dst->maxTexture2DLayered));
    memcpy(dst->maxTextureCubemapLayered, src->maxTextureCubemapLayered, sizeof(dst->maxTextureCubemapLayered));
    dst->maxSurface1D = src->maxSurface1D;
    memcpy(dst->maxSurface2D, src->maxSurface2D, sizeof(dst->maxSurface2D));
    memcpy(dst->maxSurface3D, src->maxSurface3D, sizeof(dst->maxSurface3D));
    memcpy(dst->maxSurface1DLayered, src->maxSurface1DLayered, sizeof(dst->maxSurface1DLayered));
    memcpy(dst->maxSurface2DLayered, src->maxSurface2DLayered, sizeof(dst->maxSurface2DLayered));
    dst->maxSurfaceCubemap = src->maxSurfaceCubemap;
    memcpy(dst->maxSurfaceCubemapLayered, src->maxSurfaceCubemapLayered, sizeof(dst->maxSurfaceCubemapLayered));
    dst->surfaceAlignment = src->surfaceAlignment;
    dst->concurrentKernels = src->concurrentKernels;
    dst->ECCEnabled = src->ECCEnabled;
    dst->pciBusID = src->pciBusID;
    dst->pciDeviceID = src->pciDeviceID;
    dst->pciDomainID = src->pciDomainID;
    dst->tccDriver = src->tccDriver;
    dst->asyncEngineCount = src->asyncEngineCount;
    dst->unifiedAddressing = src->unifiedAddressing;
#if CUDART_VERSION < 13000
    dst->memoryClockRate = src->memoryClockRate;
#endif
    dst->memoryBusWidth = src->memoryBusWidth;
    dst->l2CacheSize = src->l2CacheSize;
    dst->persistingL2CacheMaxSize = src->persistingL2CacheMaxSize;
    dst->maxThreadsPerMultiProcessor = src->maxThreadsPerMultiProcessor;
    dst->streamPrioritiesSupported = src->streamPrioritiesSupported;
    dst->globalL1CacheSupported = src->globalL1CacheSupported;
    dst->localL1CacheSupported = src->localL1CacheSupported;
    dst->sharedMemPerMultiprocessor = src->sharedMemPerMultiprocessor;
    dst->regsPerMultiprocessor = src->regsPerMultiprocessor;
    dst->managedMemory = src->managedMemory;
    dst->isMultiGpuBoard = src->isMultiGpuBoard;
    dst->multiGpuBoardGroupID = src->multiGpuBoardGroupID;
    dst->hostNativeAtomicSupported = src->hostNativeAtomicSupported;
#if CUDART_VERSION < 13000
    dst->singleToDoublePrecisionPerfRatio = src->singleToDoublePrecisionPerfRatio;
#endif
    dst->pageableMemoryAccess = src->pageableMemoryAccess;
    dst->concurrentManagedAccess = src->concurrentManagedAccess;
    dst->computePreemptionSupported = src->computePreemptionSupported;
    dst->canUseHostPointerForRegisteredMem = src->canUseHostPointerForRegisteredMem;
    dst->cooperativeLaunch = src->cooperativeLaunch;
#if CUDART_VERSION < 13000
    dst->cooperativeMultiDeviceLaunch = src->cooperativeMultiDeviceLaunch;
#endif
    dst->sharedMemPerBlockOptin = src->sharedMemPerBlockOptin;
    dst->pageableMemoryAccessUsesHostPageTables = src->pageableMemoryAccessUsesHostPageTables;
    dst->directManagedMemAccessFromHost = src->directManagedMemAccessFromHost;
    dst->maxBlocksPerMultiProcessor = src->maxBlocksPerMultiProcessor;
    dst->accessPolicyMaxWindowSize = src->accessPolicyMaxWindowSize;
    dst->reservedSharedMemPerBlock = src->reservedSharedMemPerBlock;
    dst->hostRegisterSupported = src->hostRegisterSupported;
    dst->sparseCudaArraySupported = src->sparseCudaArraySupported;
    dst->hostRegisterReadOnlySupported = src->hostRegisterReadOnlySupported;
    dst->timelineSemaphoreInteropSupported = src->timelineSemaphoreInteropSupported;
    dst->memoryPoolsSupported = src->memoryPoolsSupported;
    dst->gpuDirectRDMASupported = src->gpuDirectRDMASupported;
    dst->gpuDirectRDMAFlushWritesOptions = src->gpuDirectRDMAFlushWritesOptions;
    dst->gpuDirectRDMAWritesOrdering = src->gpuDirectRDMAWritesOrdering;
    dst->memoryPoolSupportedHandleTypes = src->memoryPoolSupportedHandleTypes;
    dst->deferredMappingCudaArraySupported = src->deferredMappingCudaArraySupported;
    dst->ipcEventSupported = src->ipcEventSupported;
    dst->clusterLaunch = src->clusterLaunch;
    dst->unifiedFunctionPointers = src->unifiedFunctionPointers;
}

/*
 * cudaGetDeviceProperties_v2 — the _v2 variant is what nvcc emits via
 * #define cudaGetDeviceProperties cudaGetDeviceProperties_v2
 * We export both names so legacy and modern binaries work.
 */
cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp *prop, int device) {
    HMCUDA_LOG_INFO("cudaGetDeviceProperties_v2(device=%d)", device);
    hmcuda_get_device_properties_req req = { .device = device };
    hmcuda_get_device_properties_resp resp;
    cudaError_t err = hmcuda_dispatch(HMCUDA_CMD_GET_DEVICE_PROPERTIES,
                                       &req, sizeof(req), &resp, sizeof(resp));
    if (err == cudaSuccess && prop)
        hmcuda_prop_to_cuda(&resp.prop, prop);
    return err;
}

/* Undefine the macro so we can export the original symbol name too */
#undef cudaGetDeviceProperties
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return cudaGetDeviceProperties_v2(prop, device);
}

cudaError_t cudaGetLastError() {
    cudaError_t err = g_last_error;
    g_last_error = cudaSuccess;
    return err;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    HMCUDA_LOG_INFO("cudaRuntimeGetVersion");
    RT_FWD_NOREQ_RESP(HMCUDA_CMD_RUNTIME_GET_VERSION,
        hmcuda_runtime_get_version_resp, runtimeVersion, (int), version);
}

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, const void *func) {
    HMCUDA_LOG_INFO("cudaFuncGetAttributes(func=%p)", func);
    if (!attr) return cudaErrorInvalidValue;

    auto it = g_func_map.find(func);
    if (it == g_func_map.end()) {
        HMCUDA_LOG_ERROR("cudaFuncGetAttributes: unknown function pointer %p", func);
        g_last_error = cudaErrorInvalidDeviceFunction;
        return cudaErrorInvalidDeviceFunction;
    }

    hmcuda_func_get_attributes_req req = { .func = it->second.handle };
    hmcuda_func_get_attributes_resp resp;
    cudaError_t err = hmcuda_dispatch(HMCUDA_CMD_FUNC_GET_ATTRIBUTES,
                                       &req, sizeof(req), &resp, sizeof(resp));
    if (err == cudaSuccess) {
        attr->sharedSizeBytes               = (size_t)resp.attr.sharedSizeBytes;
        attr->constSizeBytes                = (size_t)resp.attr.constSizeBytes;
        attr->localSizeBytes                = (size_t)resp.attr.localSizeBytes;
        attr->maxThreadsPerBlock            = resp.attr.maxThreadsPerBlock;
        attr->numRegs                       = resp.attr.numRegs;
        attr->ptxVersion                    = resp.attr.ptxVersion;
        attr->binaryVersion                 = resp.attr.binaryVersion;
        attr->cacheModeCA                   = resp.attr.cacheModeCA;
        attr->maxDynamicSharedSizeBytes     = resp.attr.maxDynamicSharedSizeBytes;
        attr->preferredShmemCarveout        = resp.attr.preferredShmemCarveout;
        attr->clusterDimMustBeSet           = resp.attr.clusterDimMustBeSet;
        attr->requiredClusterWidth          = resp.attr.requiredClusterWidth;
        attr->requiredClusterHeight         = resp.attr.requiredClusterHeight;
        attr->requiredClusterDepth          = resp.attr.requiredClusterDepth;
        attr->clusterSchedulingPolicyPreference = resp.attr.clusterSchedulingPolicyPreference;
        attr->nonPortableClusterSizeAllowed = resp.attr.nonPortableClusterSizeAllowed;
    }
    return err;
}

cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *hostFun) {
    HMCUDA_LOG_INFO("__cudaGetKernel(hostFun=%p)", hostFun);
    if (!kernel || !hostFun) return cudaErrorInvalidValue;

    if (g_func_map.find(hostFun) == g_func_map.end()) {
        HMCUDA_LOG_ERROR("__cudaGetKernel: unknown function pointer %p", hostFun);
        return cudaErrorInvalidDeviceFunction;
    }

    *kernel = (cudaKernel_t)hostFun;
    return cudaSuccess;
}

cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                               void **args, size_t sharedMem,
                               cudaStream_t stream) {
    return cudaLaunchKernel((const void *)kernel, gridDim, blockDim, args,
                            sharedMem, stream);
}

cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                                    void **args, size_t sharedMem,
                                    cudaStream_t stream) {
    return __cudaLaunchKernel(kernel, gridDim, blockDim, args, sharedMem, stream);
}

/* --- Internal CUDA Registration API --- */

void __cudaInitModule(void **fatCubinHandle) {
    if (!fatCubinHandle) return;
    uint64_t handle = *(uint64_t*)fatCubinHandle;
    HMCUDA_LOG_INFO("__cudaInitModule handle=0x%lx", handle);

    hmcuda_init_module_req req;
    req.fatbin_handle = handle;
    hmcuda_dispatch(HMCUDA_CMD_INIT_MODULE, &req, sizeof(req), nullptr, 0);
}

cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    hmcuda_push_call_cfg_req req = {
        .gridDim = { gridDim.x, gridDim.y, gridDim.z },
        .blockDim = { blockDim.x, blockDim.y, blockDim.z },
        .sharedMem = (uint64_t)sharedMem,
        .stream = (uint64_t)stream
    };
    RT_FWD(HMCUDA_CMD_PUSH_CALL_CONFIGURATION, req);
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, cudaStream_t *stream) {
    hmcuda_pop_call_cfg_resp resp;
    cudaError_t err = hmcuda_dispatch(HMCUDA_CMD_POP_CALL_CONFIGURATION, nullptr, 0, &resp, sizeof(resp));
    if (err == cudaSuccess) {
        if (gridDim) *gridDim = dim3(resp.gridDim.x, resp.gridDim.y, resp.gridDim.z);
        if (blockDim) *blockDim = dim3(resp.blockDim.x, resp.blockDim.y, resp.blockDim.z);
        if (sharedMem) *sharedMem = (size_t)resp.sharedMem;
        if (stream) *stream = (cudaStream_t)resp.stream;
    }
    return err;
}

struct __fatBinC_Wrapper_t {
  unsigned int magic;
  unsigned int version;
  void * data;
  void * filename_or_fatbins;
};

struct __fatBinaryHeader {
  unsigned int           magic;
  unsigned short         version;
  unsigned short         headerSize;
  unsigned long long int fatSize;
};

void** __cudaRegisterFatBinary(void *fatCubin) {
    if (!fatCubin) return NULL;

    void *fatbin_data = NULL;
    size_t size = 0;
    uint32_t magic = *(uint32_t*)fatCubin;

    if ((magic & 0xFFFF0000) == 0x46620000) {
        __fatBinC_Wrapper_t *wrapper = (__fatBinC_Wrapper_t*)fatCubin;
        fatbin_data = wrapper->data;
        HMCUDA_LOG_INFO("__cudaRegisterFatBinary: wrapper detected (magic=0x%x)", magic);
    } else {
        fatbin_data = fatCubin;
    }

    struct __fatBinaryHeader *header = (struct __fatBinaryHeader*)fatbin_data;

    if (header->magic == 0xBA55ED50 || header->magic == 0x50ED5AB4 ||
        header->magic == 0x466243b1 || header->magic == 0xB1436246) {
        size = (size_t)header->fatSize;
    } else {
        HMCUDA_LOG_ERROR("Warning: Unknown fatbin magic 0x%x, using fallback size", header->magic);
        size = 65536;
    }

    if (size == 0 || size > 100*1024*1024) {
        HMCUDA_LOG_ERROR("Warning: Invalid fatbin size %zu", size);
        size = 65536;
    }

    HMCUDA_LOG_INFO("__cudaRegisterFatBinary size=%zu (magic=0x%x)", size, header->magic);

    hmcuda_register_fatbin_req req;
    req.size = size;

    std::vector<uint8_t> payload(sizeof(req) + size);
    memcpy(payload.data(), &req, sizeof(req));
    memcpy(payload.data() + sizeof(req), fatbin_data, size);

    hmcuda_register_fatbin_resp resp;
    if (hmcuda_dispatch(HMCUDA_CMD_REGISTER_FATBIN, payload.data(), payload.size(), &resp, sizeof(resp)) != cudaSuccess) {
        return NULL;
    }

    uint64_t* handle = new uint64_t(resp.fatbin_handle);
    return (void**)handle;
}

void __cudaRegisterFunction(
        void **fatCubinHandle, const char *hostFun, char *deviceFun,
        const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    HMCUDA_LOG_INFO("__cudaRegisterFunction hostFun=%p devFun=%s", hostFun, deviceFun);
    if (!fatCubinHandle) return;

    uint64_t fatbin_handle = *(uint64_t*)fatCubinHandle;
    size_t name_len = strlen(deviceFun) + 1;

    hmcuda_register_function_req req;
    req.fatbin_handle = fatbin_handle;
    req.host_fun = (uint64_t)hostFun;
    req.device_name_len = name_len;

    std::vector<uint8_t> payload(sizeof(req) + name_len);
    memcpy(payload.data(), &req, sizeof(req));
    memcpy(payload.data() + sizeof(req), deviceFun, name_len);

    hmcuda_register_function_resp func_resp = {0, 0};
    cudaError_t err = hmcuda_dispatch(HMCUDA_CMD_REGISTER_FUNCTION, payload.data(), payload.size(), &func_resp, sizeof(func_resp));

    FuncInfo fi;
    fi.handle = req.host_fun;
    fi.param_count = count_params_from_mangled(deviceFun);
    fi.param_total_size = func_resp.param_total_size;
    g_func_map[hostFun] = fi;

    HMCUDA_LOG_INFO("Registered function: %p -> handle=0x%lx, params=%u (err=%d)",
           hostFun, fi.handle, fi.param_count, (int)err);
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    if (fatCubinHandle) delete (uint64_t*)fatCubinHandle;
}

} // extern "C"
