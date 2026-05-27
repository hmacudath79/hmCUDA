/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef HMCUDA_CMD_RUNTIME_H
#define HMCUDA_CMD_RUNTIME_H

#include "hmcuda_types.h"

/*
 * CUDA Runtime API request/response structures.
 *
 * Each pair corresponds to a HMCUDA_CMD_* command from hmcuda_types.h.
 * Structures without a _resp counterpart return only the common
 * hmcuda_resp_header (i.e. just the cuda_error code).
 */

/* cudaMalloc */
struct hmcuda_malloc_req {
    uint64_t size;
};
struct hmcuda_malloc_resp {
    uint64_t devPtr;
};

/* cudaFree */
struct hmcuda_free_req {
    uint64_t devPtr;
};

/* cudaMemcpy / cuMemcpyHtoD / cuMemcpyDtoH / cuMemcpyAsync */
struct hmcuda_memcpy_req {
    uint64_t dst;
    uint64_t src;
    uint64_t count;
    uint32_t kind;   /* cudaMemcpyKind: 1=H2D, 2=D2H */
    uint32_t _pad;   /* alignment */
    uint64_t stream; /* raw CUstream value: 0=NULL, 1=CU_STREAM_LEGACY,
                        2=CU_STREAM_PER_THREAD, other=resource handle */
};

/* cudaMemset */
struct hmcuda_memset_req {
    uint64_t devPtr;
    int32_t value;
    uint64_t count;
};

/* cudaLaunchKernel */
/* The argument blob follows this struct in the request buffer. */
struct hmcuda_launch_kernel_req {
    uint64_t func; /* Address of the kernel function */
    struct hmcuda_dim3 gridDim;
    struct hmcuda_dim3 blockDim;
    uint32_t sharedMem;
    uint64_t stream;
    uint32_t args_size; /* Total size of the kernel arguments blob */
};

/* cudaStreamCreate */
struct hmcuda_stream_create_req {
    uint32_t flags;
};
struct hmcuda_stream_create_resp {
    uint64_t stream;
};

/* cudaStreamSynchronize */
struct hmcuda_stream_synchronize_req {
    uint64_t stream;
};

/* cudaEventCreate */
struct hmcuda_event_create_req {
    uint32_t flags;
};
struct hmcuda_event_create_resp {
    uint64_t event;
};

/* cudaEventRecord */
struct hmcuda_event_record_req {
    uint64_t event;
    uint64_t stream;
};

/* cudaEventDestroy */
struct hmcuda_event_destroy_req {
    uint64_t event;
};

/* cudaEventSynchronize */
struct hmcuda_event_synchronize_req {
    uint64_t event;
};

/* cudaEventElapsedTime */
struct hmcuda_event_elapsed_time_req {
    uint64_t start_event;
    uint64_t end_event;
};
struct hmcuda_event_elapsed_time_resp {
    float ms;
};

/* cudaDeviceSynchronize */
struct hmcuda_device_synchronize_req {
    uint32_t flags;
};

/* __cudaRegisterFatBinary */
struct hmcuda_register_fatbin_req {
    uint64_t size;
    /* Data follows */
};
struct hmcuda_register_fatbin_resp {
    uint64_t fatbin_handle;
};

/* __cudaRegisterFunction */
struct hmcuda_register_function_req {
    uint64_t fatbin_handle;
    uint64_t host_fun;
    uint32_t device_name_len;
    /* Device name string follows */
};
struct hmcuda_register_function_resp {
    uint32_t param_count;
    uint32_t param_total_size;
};

/* __cudaPushCallConfiguration */
struct hmcuda_push_call_cfg_req {
    struct hmcuda_dim3 gridDim;
    struct hmcuda_dim3 blockDim;
    uint64_t sharedMem;
    uint64_t stream;
};

/* __cudaPopCallConfiguration */
struct hmcuda_pop_call_cfg_resp {
    struct hmcuda_dim3 gridDim;
    struct hmcuda_dim3 blockDim;
    uint64_t sharedMem;
    uint64_t stream;
};

/* __cudaInitModule */
struct hmcuda_init_module_req {
    uint64_t fatbin_handle;
};

/* cudaGetDeviceProperties / cudaGetDeviceProperties_v2 */
struct hmcuda_get_device_properties_req {
    int32_t device;
};

/*
 * Wire-format mirror of cudaDeviceProp.
 * Field order and sizes must match exactly so the host can memcpy
 * a real cudaDeviceProp into this struct.
 */
struct hmcuda_device_prop {
    char         name[256];
    uint8_t      uuid[16];           /* cudaUUID_t */
    char         luid[8];
    uint32_t     luidDeviceNodeMask;
    uint64_t     totalGlobalMem;     /* size_t */
    uint64_t     sharedMemPerBlock;
    int32_t      regsPerBlock;
    int32_t      warpSize;
    uint64_t     memPitch;
    int32_t      maxThreadsPerBlock;
    int32_t      maxThreadsDim[3];
    int32_t      maxGridSize[3];
    int32_t      clockRate;
    uint64_t     totalConstMem;
    int32_t      major;
    int32_t      minor;
    uint64_t     textureAlignment;
    uint64_t     texturePitchAlignment;
    int32_t      deviceOverlap;
    int32_t      multiProcessorCount;
    int32_t      kernelExecTimeoutEnabled;
    int32_t      integrated;
    int32_t      canMapHostMemory;
    int32_t      computeMode;
    int32_t      maxTexture1D;
    int32_t      maxTexture1DMipmap;
    int32_t      maxTexture1DLinear;
    int32_t      maxTexture2D[2];
    int32_t      maxTexture2DMipmap[2];
    int32_t      maxTexture2DLinear[3];
    int32_t      maxTexture2DGather[2];
    int32_t      maxTexture3D[3];
    int32_t      maxTexture3DAlt[3];
    int32_t      maxTextureCubemap;
    int32_t      maxTexture1DLayered[2];
    int32_t      maxTexture2DLayered[3];
    int32_t      maxTextureCubemapLayered[2];
    int32_t      maxSurface1D;
    int32_t      maxSurface2D[2];
    int32_t      maxSurface3D[3];
    int32_t      maxSurface1DLayered[2];
    int32_t      maxSurface2DLayered[3];
    int32_t      maxSurfaceCubemap;
    int32_t      maxSurfaceCubemapLayered[2];
    uint64_t     surfaceAlignment;
    int32_t      concurrentKernels;
    int32_t      ECCEnabled;
    int32_t      pciBusID;
    int32_t      pciDeviceID;
    int32_t      pciDomainID;
    int32_t      tccDriver;
    int32_t      asyncEngineCount;
    int32_t      unifiedAddressing;
    int32_t      memoryClockRate;
    int32_t      memoryBusWidth;
    int32_t      l2CacheSize;
    int32_t      persistingL2CacheMaxSize;
    int32_t      maxThreadsPerMultiProcessor;
    int32_t      streamPrioritiesSupported;
    int32_t      globalL1CacheSupported;
    int32_t      localL1CacheSupported;
    uint64_t     sharedMemPerMultiprocessor;
    int32_t      regsPerMultiprocessor;
    int32_t      managedMemory;
    int32_t      isMultiGpuBoard;
    int32_t      multiGpuBoardGroupID;
    int32_t      hostNativeAtomicSupported;
    int32_t      singleToDoublePrecisionPerfRatio;
    int32_t      pageableMemoryAccess;
    int32_t      concurrentManagedAccess;
    int32_t      computePreemptionSupported;
    int32_t      canUseHostPointerForRegisteredMem;
    int32_t      cooperativeLaunch;
    int32_t      cooperativeMultiDeviceLaunch;
    uint64_t     sharedMemPerBlockOptin;
    int32_t      pageableMemoryAccessUsesHostPageTables;
    int32_t      directManagedMemAccessFromHost;
    int32_t      maxBlocksPerMultiProcessor;
    int32_t      accessPolicyMaxWindowSize;
    uint64_t     reservedSharedMemPerBlock;
    int32_t      hostRegisterSupported;
    int32_t      sparseCudaArraySupported;
    int32_t      hostRegisterReadOnlySupported;
    int32_t      timelineSemaphoreInteropSupported;
    int32_t      memoryPoolsSupported;
    int32_t      gpuDirectRDMASupported;
    uint32_t     gpuDirectRDMAFlushWritesOptions;
    int32_t      gpuDirectRDMAWritesOrdering;
    uint32_t     memoryPoolSupportedHandleTypes;
    int32_t      deferredMappingCudaArraySupported;
    int32_t      ipcEventSupported;
    int32_t      clusterLaunch;
    int32_t      unifiedFunctionPointers;
    int32_t      reserved2[2];
    int32_t      reserved[61];
};

struct hmcuda_get_device_properties_resp {
    struct hmcuda_device_prop prop;
};

/* cudaGetDeviceCount */
struct hmcuda_get_device_count_resp {
    int32_t count;
};

/* cudaSetDevice */
struct hmcuda_set_device_req {
    int32_t device;
};

/* cudaGetErrorString / cudaGetErrorName */
struct hmcuda_get_error_string_req {
    uint32_t error;
};
struct hmcuda_get_error_string_resp {
    uint32_t str_len;
    /* String data follows */
};

/* cudaRuntimeGetVersion */
struct hmcuda_runtime_get_version_resp {
    int32_t version;
};

/* cudaFuncGetAttributes */
struct hmcuda_func_get_attributes_req {
    uint64_t func; /* function handle from g_func_map */
};
/* Wire-format mirror of cudaFuncAttributes. */
struct hmcuda_func_attributes {
    uint64_t sharedSizeBytes;
    uint64_t constSizeBytes;
    uint64_t localSizeBytes;
    int32_t  maxThreadsPerBlock;
    int32_t  numRegs;
    int32_t  ptxVersion;
    int32_t  binaryVersion;
    int32_t  cacheModeCA;
    int32_t  maxDynamicSharedSizeBytes;
    int32_t  preferredShmemCarveout;
    int32_t  clusterDimMustBeSet;
    int32_t  requiredClusterWidth;
    int32_t  requiredClusterHeight;
    int32_t  requiredClusterDepth;
    int32_t  clusterSchedulingPolicyPreference;
    int32_t  nonPortableClusterSizeAllowed;
};
struct hmcuda_func_get_attributes_resp {
    struct hmcuda_func_attributes attr;
};

#endif /* HMCUDA_CMD_RUNTIME_H */
