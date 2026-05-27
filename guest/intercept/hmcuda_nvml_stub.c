/*
 * Minimal NVML stub for hmCUDA guest environment.
 * Returns success / dummy values for the functions nvbandwidth uses.
 */

#include <string.h>

/* Mirror the types/enums nvbandwidth needs */
typedef enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_NOT_SUPPORTED = 3,
} nvmlReturn_t;

typedef struct nvmlDevice_st *nvmlDevice_t;

#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE 80

nvmlReturn_t nvmlInit(void)
{
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlInit_v2(void)
{
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlShutdown(void)
{
    return NVML_SUCCESS;
}

const char *nvmlErrorString(nvmlReturn_t result)
{
    if (result == NVML_SUCCESS)
        return "Success";
    return "hmCUDA NVML stub: not supported";
}

nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)
{
    const char *ver = "hmCUDA-stub";
    if (version && length > 0) {
        strncpy(version, ver, length);
        version[length - 1] = '\0';
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device)
{
    if (device)
        *device = (nvmlDevice_t)(void *)0x1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device)
{
    (void)device;
    return NVML_ERROR_NOT_SUPPORTED;
}
