/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef HMCUDA_ABI_H
#define HMCUDA_ABI_H

/*
 * Umbrella header for the hmCUDA ABI.
 *
 * Includes all command definitions. Individual headers can also be
 * included directly when only a subset is needed.
 *
 * To add a new library (e.g. cuBLAS):
 *   1. Reserve a command ID range in hmcuda_types.h
 *   2. Create common/include/hmcuda_cmd_cublas.h with request/response structs
 *   3. Include it here
 */

#include "hmcuda_types.h"
#include "hmcuda_cmd_runtime.h"
#include "hmcuda_cmd_driver.h"

/* Future libraries:
 * #include "hmcuda_cmd_cublas.h"
 * #include "hmcuda_cmd_cudnn.h"
 */

#endif /* HMCUDA_ABI_H */
