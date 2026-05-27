#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "hmcuda_api.h"
#include "resource.h"
#include "log.h"

/* Global log level - can be set via environment variable */
LogLevel g_log_level = VHOST_LOG_LEVEL;

/* Internal CUDA runtime functions */
extern cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
extern cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, cudaStream_t *stream);
extern void __cudaInitModule(void **fatCubinHandle);

static CUstream resolve_runtime_stream(Context *ctx, uint64_t stream_handle)
{
    if (stream_handle == 0)
        return NULL;
    if (stream_handle == (uint64_t)(uintptr_t)CU_STREAM_LEGACY)
        return CU_STREAM_LEGACY;
    if (stream_handle == (uint64_t)(uintptr_t)CU_STREAM_PER_THREAD)
        return CU_STREAM_PER_THREAD;
    CUstream stream = (CUstream)hmcuda_res_get(ctx, RES_TYPE_STREAM, stream_handle);
    return stream ? stream : NULL;
}

uint32_t hmcuda_core_init(uint32_t session_id, uint32_t vm_id)
{
    LOG("hmcuda_core_init: session_id=%u, vm_id=%u", session_id, vm_id);

    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (ctx) {
        hmcuda_ctx_destroy(ctx);
    }
    hmcuda_ctx_create(session_id, vm_id);

    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        const char *err_str;
        cuGetErrorString(res, &err_str);
        LOG_ERROR("hmcuda_core_init: cuInit failed: %s", err_str);
        return cudaErrorInitializationError;
    }

    /* Select device 0. If a previous session left the primary context in a
     * sticky error state, cudaDeviceReset() clears it before re-initialising. */
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return err;
    }

    /* cudaFree(0) forces full context initialisation.  If it fails with a
     * sticky error from a previous session, reset and retry once. */
    err = cudaFree(0);
    if (err != cudaSuccess) {
        LOG("hmcuda_core_init: context in error state (%d), resetting device", err);
        cudaDeviceReset();
        err = cudaSetDevice(0);
        if (err != cudaSuccess) return err;
        err = cudaFree(0);
    }
    return err;
}

uint32_t hmcuda_core_malloc(uint32_t session_id, uint32_t vm_id, uint64_t size, uint64_t *devPtr)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, size);
    
    if (err == cudaSuccess) {
        /* Use the real device VA as the handle so pointers stored inside
         * device memory (e.g. linked lists) contain dereferenceable addresses. */
        *devPtr = (uint64_t)(uintptr_t)ptr;
        hmcuda_res_add_with_handle(ctx, RES_TYPE_MEM, ptr, size, *devPtr);
    } else {
        *devPtr = 0;
    }
    LOG("hmcuda_core_malloc: size=%lu, handle=0x%lx, real_ptr=%p, err=%d", size, *devPtr, ptr, err);
    return err;
}

uint32_t hmcuda_core_free(uint32_t session_id, uint32_t vm_id, uint64_t devPtr)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    void *real_ptr = hmcuda_res_get(ctx, RES_TYPE_MEM, devPtr);
    LOG("hmcuda_core_free: handle=0x%lx, real_ptr=%p", devPtr, real_ptr);
    
    if (real_ptr) {
        cudaError_t err = cudaFree(real_ptr);
        hmcuda_res_remove(ctx, RES_TYPE_MEM, devPtr);
        return err;
    }
    return cudaErrorInvalidValue;
}

uint32_t hmcuda_core_memcpy(uint32_t session_id, uint32_t vm_id, uint64_t dst, uint64_t src,
                            uint64_t count, uint32_t kind, void *host_ptr, uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;

    CUstream stream = resolve_runtime_stream(ctx, stream_handle);

    void *real_dst = NULL;
    void *real_src = NULL;

    if (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice) {
        real_dst = hmcuda_res_get(ctx, RES_TYPE_MEM, dst);
        if (!real_dst) {
            LOG("hmcuda_core_memcpy H2D/D2D: Failed to find resource for handle 0x%lx", dst);
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice) {
        real_src = hmcuda_res_get(ctx, RES_TYPE_MEM, src);
        if (!real_src) {
            LOG("hmcuda_core_memcpy D2H/D2D: Failed to find resource for handle 0x%lx", src);
            return CUDA_ERROR_INVALID_VALUE;
        }
    }

    LOG("hmcuda_core_memcpy: dst_handle=0x%lx, src_handle=0x%lx, real_dst=%p, real_src=%p, count=%lu, kind=%u, stream=%p",
           dst, src, real_dst, real_src, count, kind, (void *)stream);

    if (kind == cudaMemcpyHostToDevice) {
        if (!host_ptr) return CUDA_ERROR_INVALID_VALUE;
        if (hmcuda_buffer_needs_host_ptr_patch(ctx, host_ptr, (size_t)count)) {
            void *patch_buf = malloc((size_t)count);
            if (!patch_buf)
                return CUDA_ERROR_OUT_OF_MEMORY;
            memcpy(patch_buf, host_ptr, (size_t)count);
            hmcuda_patch_embedded_host_ptrs(ctx, patch_buf, (size_t)count);
            CUresult res = cuMemcpyHtoDAsync((CUdeviceptr)real_dst, patch_buf, (size_t)count, stream);
            if (res == CUDA_SUCCESS) res = cuStreamSynchronize(stream);
            free(patch_buf);
            return res;
        }
        CUresult res = cuMemcpyHtoDAsync((CUdeviceptr)real_dst, host_ptr, (size_t)count, stream);
        if (res == CUDA_SUCCESS) res = cuStreamSynchronize(stream);
        return res;
    } else if (kind == cudaMemcpyDeviceToHost) {
        if (!host_ptr) return CUDA_ERROR_INVALID_VALUE;
        return cuMemcpyDtoHAsync(host_ptr, (CUdeviceptr)real_src, (size_t)count, stream);
    } else if (kind == cudaMemcpyDeviceToDevice) {
        return cuMemcpyAsync((CUdeviceptr)real_dst, (CUdeviceptr)real_src,
                             (size_t)count, stream);
    }
    return CUDA_ERROR_INVALID_VALUE;
}

uint32_t hmcuda_core_runtime_memcpy_htod_vq(uint32_t session_id, uint32_t vm_id,
                                            uint64_t dst, void *host_ptr,
                                            uint64_t count, uint64_t stream_handle)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return CUDA_ERROR_NOT_INITIALIZED;
    if (!host_ptr) return CUDA_ERROR_INVALID_VALUE;

    CUstream stream = resolve_runtime_stream(ctx, stream_handle);
    void *real_dst = hmcuda_res_get(ctx, RES_TYPE_MEM, dst);
    if (!real_dst) {
        LOG("hmcuda_core_runtime_memcpy_htod_vq: failed to resolve dst=0x%lx", dst);
        return CUDA_ERROR_INVALID_VALUE;
    }

    return cuMemcpyHtoDAsync((CUdeviceptr)real_dst, host_ptr, (size_t)count, stream);
}

uint32_t hmcuda_core_memset(uint32_t session_id, uint32_t vm_id, uint64_t devPtr, int32_t value, uint64_t count)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    // hmcuda_res_get already handles handle+offset for memory resources
    void *real_ptr = hmcuda_res_get(ctx, RES_TYPE_MEM, devPtr);
    if (!real_ptr) {
        LOG("hmcuda_core_memset: Failed to find resource for handle 0x%lx", devPtr);
        return cudaErrorInvalidValue;
    }

    LOG("hmcuda_core_memset: handle=0x%lx, real_ptr=%p, value=%d, count=%lu",
           devPtr, real_ptr, value, count);

    return cudaMemset(real_ptr, value, count);
}

uint32_t hmcuda_core_launch_kernel(uint32_t session_id, uint32_t vm_id, uint64_t func, struct hmcuda_dim3 gridDim, struct hmcuda_dim3 blockDim, uint32_t sharedMem, uint64_t stream, void *args, uint32_t args_size)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    CUfunction real_func = (CUfunction)hmcuda_res_get(ctx, RES_TYPE_FUNCTION, func);
    CUstream real_stream = resolve_runtime_stream(ctx, stream);

    LOG("hmcuda_core_launch_kernel: func_handle=0x%lx, real_func=%p, grid(%u,%u,%u), block(%u,%u,%u), shmem=%u, stream_handle=%lu, real_stream=%p, args_size=%u",
           func, real_func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, real_stream, args_size);

    if (!real_func) {
        LOG_ERROR("hmcuda_core_launch_kernel: Invalid function handle");
        return cudaErrorInvalidValue;
    }

    CUresult res;
    if (args && args_size > 0) {
        // Guest packs args as 8 bytes each in a flat buffer.
        // Translate any device pointer handles to real pointers.
        uint32_t param_count = args_size / 8;
        uint64_t *args_u64 = (uint64_t *)args;

        LOG("hmcuda_core_launch_kernel: param_count=%u, args_size=%u", param_count, args_size);

        // Translate device pointer handles to real pointers
        for (uint32_t i = 0; i < param_count; i++) {
            uint64_t handle = args_u64[i];
            void *real_ptr = hmcuda_res_get(ctx, RES_TYPE_MEM, handle);
            if (!real_ptr)
                real_ptr = hmcuda_res_get(ctx, RES_TYPE_HOST_MEM, handle);
            if (real_ptr) {
                LOG("hmcuda_core_launch_kernel: arg[%u]: translating handle 0x%lx -> real_ptr %p", i, handle, real_ptr);
                args_u64[i] = (uint64_t)real_ptr;
            } else {
                LOG("hmcuda_core_launch_kernel: arg[%u]: value 0x%lx (not a device pointer)", i, handle);
            }
        }

        // Reconstruct kernelParams: array of pointers into the packed buffer.
        void **kernelParams = (void **)malloc(param_count * sizeof(void *));
        for (uint32_t i = 0; i < param_count; i++) {
            kernelParams[i] = (char *)args + i * 8;
        }

        res = cuLaunchKernel(
            real_func,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            sharedMem,
            real_stream,
            kernelParams,
            NULL
        );
        free(kernelParams);
    } else {
        res = cuLaunchKernel(
            real_func,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            sharedMem,
            real_stream,
            NULL,
            NULL
        );
    }

    if (res != CUDA_SUCCESS) {
        const char *err_str;
        cuGetErrorString(res, &err_str);
        LOG_ERROR("hmcuda_core_launch_kernel: cuLaunchKernel failed: %s", err_str);
        return cudaErrorLaunchFailure;
    }

    LOG("hmcuda_core_launch_kernel: cuLaunchKernel succeeded");
    return cudaSuccess;
}

uint32_t hmcuda_core_stream_create(uint32_t session_id, uint32_t vm_id, uint32_t flags, uint64_t *stream)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    cudaStream_t s;
    cudaError_t err = cudaStreamCreateWithFlags(&s, flags);
    
    if (err == cudaSuccess) {
        *stream = hmcuda_res_add(ctx, RES_TYPE_STREAM, s, 0);
    } else {
        *stream = 0;
    }
    LOG("hmcuda_core_stream_create: flags=%u, handle=%lu, real_stream=%p, err=%d", flags, *stream, s, err);
    return err;
}

uint32_t hmcuda_core_stream_synchronize(uint32_t session_id, uint32_t vm_id, uint64_t stream)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    cudaStream_t real_stream = (cudaStream_t)resolve_runtime_stream(ctx, stream);
    LOG("hmcuda_core_stream_synchronize: handle=%lu, real_stream=%p", stream, real_stream);
    return cudaStreamSynchronize(real_stream);
}

uint32_t hmcuda_core_event_create(uint32_t session_id, uint32_t vm_id, uint32_t flags, uint64_t *event)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    cudaEvent_t e;
    cudaError_t err = cudaEventCreateWithFlags(&e, flags);
    
    if (err == cudaSuccess) {
        *event = hmcuda_res_add(ctx, RES_TYPE_EVENT, e, 0);
    } else {
        *event = 0;
    }
    LOG("hmcuda_core_event_create: flags=%u, handle=%lu, real_event=%p, err=%d", flags, *event, e, err);
    return err;
}

uint32_t hmcuda_core_event_record(uint32_t session_id, uint32_t vm_id, uint64_t event, uint64_t stream)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    cudaEvent_t real_event = (cudaEvent_t)hmcuda_res_get(ctx, RES_TYPE_EVENT, event);
    cudaStream_t real_stream = (cudaStream_t)resolve_runtime_stream(ctx, stream);
    
    LOG("hmcuda_core_event_record: event_handle=%lu, stream_handle=%lu", event, stream);
    return cudaEventRecord(real_event, real_stream);
}

uint32_t hmcuda_core_event_synchronize(uint32_t session_id, uint32_t vm_id, uint64_t event)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    cudaEvent_t real_event = (cudaEvent_t)hmcuda_res_get(ctx, RES_TYPE_EVENT, event);
    LOG("hmcuda_core_event_synchronize: handle=%lu", event);
    return cudaEventSynchronize(real_event);
}

uint32_t hmcuda_core_event_elapsed_time(uint32_t session_id, uint32_t vm_id, uint64_t start, uint64_t end, float *ms)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    cudaEvent_t real_start = (cudaEvent_t)hmcuda_res_get(ctx, RES_TYPE_EVENT, start);
    cudaEvent_t real_end = (cudaEvent_t)hmcuda_res_get(ctx, RES_TYPE_EVENT, end);
    
    LOG("hmcuda_core_event_elapsed_time: start_handle=%lu, end_handle=%lu", start, end);
    return cudaEventElapsedTime(ms, real_start, real_end);
}

uint32_t hmcuda_core_event_destroy(uint32_t session_id, uint32_t vm_id, uint64_t event)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    cudaEvent_t real_event = (cudaEvent_t)hmcuda_res_get(ctx, RES_TYPE_EVENT, event);
    LOG("hmcuda_core_event_destroy: handle=%lu", event);
    
    if (real_event) {
        cudaError_t err = cudaEventDestroy(real_event);
        hmcuda_res_remove(ctx, RES_TYPE_EVENT, event);
        return err;
    }
    return cudaErrorInvalidValue;
}

uint32_t hmcuda_core_device_synchronize(uint32_t session_id, uint32_t vm_id)
{
    LOG("hmcuda_core_device_synchronize");
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR("hmcuda_core_device_synchronize: cudaDeviceSynchronize failed: %s", cudaGetErrorString(err));
    } else {
        LOG("hmcuda_core_device_synchronize: succeeded");
    }
    return err;
}

uint32_t hmcuda_core_push_call_configuration(uint32_t session_id, uint32_t vm_id, struct hmcuda_dim3 gridDim, struct hmcuda_dim3 blockDim, uint64_t sharedMem, uint64_t stream)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    cudaStream_t real_stream = (cudaStream_t)resolve_runtime_stream(ctx, stream);
    dim3 g = { gridDim.x, gridDim.y, gridDim.z };
    dim3 b = { blockDim.x, blockDim.y, blockDim.z };

    return __cudaPushCallConfiguration(g, b, (size_t)sharedMem, real_stream);
}

uint32_t hmcuda_core_pop_call_configuration(uint32_t session_id, uint32_t vm_id, struct hmcuda_dim3 *gridDim, struct hmcuda_dim3 *blockDim, uint64_t *sharedMem, uint64_t *stream)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    dim3 g, b;
    size_t s_mem;
    cudaStream_t s;

    cudaError_t err = __cudaPopCallConfiguration(&g, &b, &s_mem, &s);
    if (err == cudaSuccess) {
        gridDim->x = g.x; gridDim->y = g.y; gridDim->z = g.z;
        blockDim->x = b.x; blockDim->y = b.y; blockDim->z = b.z;
        *sharedMem = (uint64_t)s_mem;
        if (s == 0) {
            *stream = 0;
        } else if (s == CU_STREAM_LEGACY) {
            *stream = (uint64_t)(uintptr_t)CU_STREAM_LEGACY;
        } else if (s == CU_STREAM_PER_THREAD) {
            *stream = (uint64_t)(uintptr_t)CU_STREAM_PER_THREAD;
        } else {
            *stream = hmcuda_res_get_handle(ctx, RES_TYPE_STREAM, s);
        }
    }
    return err;
}

uint32_t hmcuda_core_register_fatbin(uint32_t session_id, uint32_t vm_id, void *fatbin_data, uint64_t size, uint64_t *fatbin_handle)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    LOG("hmcuda_core_register_fatbin: size=%lu", size);

    // Use CUDA Driver API to load the module
    CUmodule module;
    CUresult res = cuModuleLoadData(&module, fatbin_data);

    if (res != CUDA_SUCCESS) {
        const char *err_str;
        cuGetErrorString(res, &err_str);
        LOG_ERROR("hmcuda_core_register_fatbin: cuModuleLoadData failed: %s", err_str);
        return cudaErrorInitializationError;
    }

    *fatbin_handle = hmcuda_res_add(ctx, RES_TYPE_MODULE, (void *)module, 0);
    LOG("hmcuda_core_register_fatbin: module=%p, handle=0x%lx", (void *)module, *fatbin_handle);
    return 0;
}

uint32_t hmcuda_core_register_function(uint32_t session_id, uint32_t vm_id, uint64_t fatbin_handle, uint64_t host_fun, const char *device_name, uint32_t *out_param_count, uint32_t *out_param_total_size)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    *out_param_count = 0;
    *out_param_total_size = 0;

    CUmodule module = (CUmodule)hmcuda_res_get(ctx, RES_TYPE_MODULE, fatbin_handle);
    if (!module) {
        LOG_ERROR("hmcuda_core_register_function: Invalid fatbin_handle");
        return cudaErrorInvalidValue;
    }

    LOG("hmcuda_core_register_function: module=%p, device_name=%s, host_fun=%p",
           (void *)module, device_name, (void *)host_fun);

    CUfunction func;
    CUresult res = cuModuleGetFunction(&func, module, device_name);

    if (res != CUDA_SUCCESS) {
        const char *err_str;
        cuGetErrorString(res, &err_str);
        LOG_ERROR("hmcuda_core_register_function: cuModuleGetFunction failed for '%s': %s",
                device_name, err_str);
        return cudaErrorInvalidValue;
    }

    // Store the function with host_fun as the handle so guest can look it up
    hmcuda_res_add_with_handle(ctx, RES_TYPE_FUNCTION, (void *)func, 0, host_fun);

    // cuFuncGetParamInfo is not available in the CUDA toolkit versions we target.
    // Param count is determined by the guest from the stub function
    *out_param_count = 0;
    *out_param_total_size = 0;

    LOG("hmcuda_core_register_function: func=%p, handle=0x%lx",
           (void *)func, host_fun);
    return 0;
}

uint32_t hmcuda_core_init_module(uint32_t session_id, uint32_t vm_id, uint64_t fatbin_handle)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    CUmodule module = (CUmodule)hmcuda_res_get(ctx, RES_TYPE_MODULE, fatbin_handle);
    LOG("hmcuda_core_init_module: handle=0x%lx, module=%p", fatbin_handle, (void *)module);

    if (!module) {
        LOG_ERROR("hmcuda_core_init_module: Invalid module handle");
        return cudaErrorInvalidValue;
    }

    // Module is already loaded, nothing more to do
    return 0;
}

uint32_t hmcuda_core_get_device_properties(uint32_t session_id, uint32_t vm_id,
                                            int device, struct hmcuda_device_prop *out)
{
    LOG("hmcuda_core_get_device_properties: device=%d", device);
    struct cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaGetDeviceProperties failed: %s", cudaGetErrorString(err));
        return err;
    }
    /* Copy field-by-field to ensure ABI match across different compilers/padding */
    memset(out, 0, sizeof(*out));
    memcpy(out->name, prop.name, sizeof(out->name));
    memcpy(out->uuid, &prop.uuid, 16);
    memcpy(out->luid, prop.luid, 8);
    out->luidDeviceNodeMask = prop.luidDeviceNodeMask;
    out->totalGlobalMem = prop.totalGlobalMem;
    out->sharedMemPerBlock = prop.sharedMemPerBlock;
    out->regsPerBlock = prop.regsPerBlock;
    out->warpSize = prop.warpSize;
    out->memPitch = prop.memPitch;
    out->maxThreadsPerBlock = prop.maxThreadsPerBlock;
    memcpy(out->maxThreadsDim, prop.maxThreadsDim, sizeof(out->maxThreadsDim));
    memcpy(out->maxGridSize, prop.maxGridSize, sizeof(out->maxGridSize));
#if CUDART_VERSION < 13000
    out->clockRate = prop.clockRate;
#endif
    out->totalConstMem = prop.totalConstMem;
    out->major = prop.major;
    out->minor = prop.minor;
    out->textureAlignment = prop.textureAlignment;
    out->texturePitchAlignment = prop.texturePitchAlignment;
#if CUDART_VERSION < 13000
    out->deviceOverlap = prop.deviceOverlap;
#endif
    out->multiProcessorCount = prop.multiProcessorCount;
#if CUDART_VERSION < 13000
    out->kernelExecTimeoutEnabled = prop.kernelExecTimeoutEnabled;
#endif
    out->integrated = prop.integrated;
    out->canMapHostMemory = prop.canMapHostMemory;
#if CUDART_VERSION < 13000
    out->computeMode = prop.computeMode;
#endif
    out->maxTexture1D = prop.maxTexture1D;
    out->maxTexture1DMipmap = prop.maxTexture1DMipmap;
#if CUDART_VERSION < 13000
    out->maxTexture1DLinear = prop.maxTexture1DLinear;
#endif
    memcpy(out->maxTexture2D, prop.maxTexture2D, sizeof(out->maxTexture2D));
    memcpy(out->maxTexture2DMipmap, prop.maxTexture2DMipmap, sizeof(out->maxTexture2DMipmap));
    memcpy(out->maxTexture2DLinear, prop.maxTexture2DLinear, sizeof(out->maxTexture2DLinear));
    memcpy(out->maxTexture2DGather, prop.maxTexture2DGather, sizeof(out->maxTexture2DGather));
    memcpy(out->maxTexture3D, prop.maxTexture3D, sizeof(out->maxTexture3D));
    memcpy(out->maxTexture3DAlt, prop.maxTexture3DAlt, sizeof(out->maxTexture3DAlt));
    out->maxTextureCubemap = prop.maxTextureCubemap;
    memcpy(out->maxTexture1DLayered, prop.maxTexture1DLayered, sizeof(out->maxTexture1DLayered));
    memcpy(out->maxTexture2DLayered, prop.maxTexture2DLayered, sizeof(out->maxTexture2DLayered));
    memcpy(out->maxTextureCubemapLayered, prop.maxTextureCubemapLayered, sizeof(out->maxTextureCubemapLayered));
    out->maxSurface1D = prop.maxSurface1D;
    memcpy(out->maxSurface2D, prop.maxSurface2D, sizeof(out->maxSurface2D));
    memcpy(out->maxSurface3D, prop.maxSurface3D, sizeof(out->maxSurface3D));
    memcpy(out->maxSurface1DLayered, prop.maxSurface1DLayered, sizeof(out->maxSurface1DLayered));
    memcpy(out->maxSurface2DLayered, prop.maxSurface2DLayered, sizeof(out->maxSurface2DLayered));
    out->maxSurfaceCubemap = prop.maxSurfaceCubemap;
    memcpy(out->maxSurfaceCubemapLayered, prop.maxSurfaceCubemapLayered, sizeof(out->maxSurfaceCubemapLayered));
    out->surfaceAlignment = prop.surfaceAlignment;
    out->concurrentKernels = prop.concurrentKernels;
    out->ECCEnabled = prop.ECCEnabled;
    out->pciBusID = prop.pciBusID;
    out->pciDeviceID = prop.pciDeviceID;
    out->pciDomainID = prop.pciDomainID;
    out->tccDriver = prop.tccDriver;
    out->asyncEngineCount = prop.asyncEngineCount;
    out->unifiedAddressing = prop.unifiedAddressing;
#if CUDART_VERSION < 13000
    out->memoryClockRate = prop.memoryClockRate;
#endif
    out->memoryBusWidth = prop.memoryBusWidth;
    out->l2CacheSize = prop.l2CacheSize;
    out->persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize;
    out->maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    out->streamPrioritiesSupported = prop.streamPrioritiesSupported;
    out->globalL1CacheSupported = prop.globalL1CacheSupported;
    out->localL1CacheSupported = prop.localL1CacheSupported;
    out->sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
    out->regsPerMultiprocessor = prop.regsPerMultiprocessor;
    out->managedMemory = prop.managedMemory;
    out->isMultiGpuBoard = prop.isMultiGpuBoard;
    out->multiGpuBoardGroupID = prop.multiGpuBoardGroupID;
    out->hostNativeAtomicSupported = prop.hostNativeAtomicSupported;
#if CUDART_VERSION < 13000
    out->singleToDoublePrecisionPerfRatio = prop.singleToDoublePrecisionPerfRatio;
#endif
    out->pageableMemoryAccess = prop.pageableMemoryAccess;
    out->concurrentManagedAccess = prop.concurrentManagedAccess;
    out->computePreemptionSupported = prop.computePreemptionSupported;
    out->canUseHostPointerForRegisteredMem = prop.canUseHostPointerForRegisteredMem;
    out->cooperativeLaunch = prop.cooperativeLaunch;
#if CUDART_VERSION < 13000
    out->cooperativeMultiDeviceLaunch = prop.cooperativeMultiDeviceLaunch;
#endif
    out->sharedMemPerBlockOptin = prop.sharedMemPerBlockOptin;
    out->pageableMemoryAccessUsesHostPageTables = prop.pageableMemoryAccessUsesHostPageTables;
    out->directManagedMemAccessFromHost = prop.directManagedMemAccessFromHost;
    out->maxBlocksPerMultiProcessor = prop.maxBlocksPerMultiProcessor;
    out->accessPolicyMaxWindowSize = prop.accessPolicyMaxWindowSize;
    out->reservedSharedMemPerBlock = prop.reservedSharedMemPerBlock;
    out->hostRegisterSupported = prop.hostRegisterSupported;
    out->sparseCudaArraySupported = prop.sparseCudaArraySupported;
    out->hostRegisterReadOnlySupported = prop.hostRegisterReadOnlySupported;
    out->timelineSemaphoreInteropSupported = prop.timelineSemaphoreInteropSupported;
    out->memoryPoolsSupported = prop.memoryPoolsSupported;
    out->gpuDirectRDMASupported = prop.gpuDirectRDMASupported;
    out->gpuDirectRDMAFlushWritesOptions = prop.gpuDirectRDMAFlushWritesOptions;
    out->gpuDirectRDMAWritesOrdering = prop.gpuDirectRDMAWritesOrdering;
    out->memoryPoolSupportedHandleTypes = prop.memoryPoolSupportedHandleTypes;
    out->deferredMappingCudaArraySupported = prop.deferredMappingCudaArraySupported;
    out->ipcEventSupported = prop.ipcEventSupported;
    out->clusterLaunch = prop.clusterLaunch;
    out->unifiedFunctionPointers = prop.unifiedFunctionPointers;
    return 0;
}

uint32_t hmcuda_core_get_device_count(uint32_t session_id, uint32_t vm_id, int *count)
{
    LOG("hmcuda_core_get_device_count");
    cudaError_t err = cudaGetDeviceCount(count);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaGetDeviceCount failed: %s", cudaGetErrorString(err));
    }
    return err;
}

uint32_t hmcuda_core_set_device(uint32_t session_id, uint32_t vm_id, int device)
{
    LOG("hmcuda_core_set_device: device=%d", device);
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaSetDevice failed: %s", cudaGetErrorString(err));
    }
    return err;
}

uint32_t hmcuda_core_get_error_string(uint32_t error, const char **str)
{
    *str = cudaGetErrorString((cudaError_t)error);
    return 0; /* Always succeeds */
}

uint32_t hmcuda_core_get_error_name(uint32_t error, const char **str)
{
    *str = cudaGetErrorName((cudaError_t)error);
    return 0; /* Always succeeds */
}

uint32_t hmcuda_core_runtime_get_version(int *version)
{
    LOG("hmcuda_core_runtime_get_version");
    cudaError_t err = cudaRuntimeGetVersion(version);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaRuntimeGetVersion failed: %s", cudaGetErrorString(err));
    }
    return err;
}

uint32_t hmcuda_core_func_get_attributes(uint32_t session_id, uint32_t vm_id,
                                          uint64_t func_handle,
                                          struct hmcuda_func_attributes *out)
{
    Context *ctx = hmcuda_ctx_get(session_id, vm_id);
    if (!ctx) return 999;

    CUfunction real_func = (CUfunction)hmcuda_res_get(ctx, RES_TYPE_FUNCTION, func_handle);
    if (!real_func) {
        LOG_ERROR("hmcuda_core_func_get_attributes: invalid function handle 0x%lx", func_handle);
        return cudaErrorInvalidDeviceFunction;
    }

    LOG("hmcuda_core_func_get_attributes: handle=0x%lx, real_func=%p", func_handle, real_func);

    /* Query attributes via the Driver API */
    memset(out, 0, sizeof(*out));

    int val;
    CUresult res;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, real_func);
    if (res == CUDA_SUCCESS) out->sharedSizeBytes = (uint64_t)val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, real_func);
    if (res == CUDA_SUCCESS) out->constSizeBytes = (uint64_t)val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, real_func);
    if (res == CUDA_SUCCESS) out->localSizeBytes = (uint64_t)val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, real_func);
    if (res == CUDA_SUCCESS) out->maxThreadsPerBlock = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_NUM_REGS, real_func);
    if (res == CUDA_SUCCESS) out->numRegs = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_PTX_VERSION, real_func);
    if (res == CUDA_SUCCESS) out->ptxVersion = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_BINARY_VERSION, real_func);
    if (res == CUDA_SUCCESS) out->binaryVersion = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, real_func);
    if (res == CUDA_SUCCESS) out->cacheModeCA = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, real_func);
    if (res == CUDA_SUCCESS) out->maxDynamicSharedSizeBytes = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, real_func);
    if (res == CUDA_SUCCESS) out->preferredShmemCarveout = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, real_func);
    if (res == CUDA_SUCCESS) out->clusterDimMustBeSet = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, real_func);
    if (res == CUDA_SUCCESS) out->requiredClusterWidth = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, real_func);
    if (res == CUDA_SUCCESS) out->requiredClusterHeight = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, real_func);
    if (res == CUDA_SUCCESS) out->requiredClusterDepth = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, real_func);
    if (res == CUDA_SUCCESS) out->clusterSchedulingPolicyPreference = val;

    res = cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, real_func);
    if (res == CUDA_SUCCESS) out->nonPortableClusterSizeAllowed = val;

    return cudaSuccess;
}
