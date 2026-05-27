#ifndef HMCUDA_TRANSPORT_H
#define HMCUDA_TRANSPORT_H

#include <cstdint>
#include <cstddef>

/*
 * Shared transport layer for hmCUDA guest libraries.
 *
 * Both libcudart.so (runtime API) and libcuda.so (driver API)
 * compile this in. Each library opens the device independently;
 * the kernel driver and vhost backend handle concurrent fds.
 */

/* Transport initialization — called by each library's constructor.
 * Opens the device and reads env vars. Safe to call multiple times. */
void hmcuda_transport_init();

/* Transport cleanup — called by each library's destructor. */
void hmcuda_transport_fini();

/* Returns true if transport is ready (device fd is open). */
bool hmcuda_transport_ready();

/*
 * Dispatch a command through the virtio (or RPC) transport.
 *
 * Returns the cuda_error field from the response header (0 = success).
 * If resp/resp_len are provided, copies the response payload.
 */
uint32_t hmcuda_transport_dispatch(uint32_t cmd,
                                   const void *req, size_t req_len,
                                   void *resp, size_t resp_len);

/*
 * Host-memory (cuMemHostAlloc) registry — shared between libcuda.so and
 * libcudart.so via hmcuda_transport.
 *
 * cuMemHostAlloc registers guest_va → alloc_size.
 * cudaLaunchKernel / cuLaunchKernel call hmcuda_transport_sync_host_mem_arg
 * before dispatch to push CPU-written content to the backend's real_ptr.
 */
void hmcuda_transport_host_mem_register(const void *guest_va, size_t size);
void hmcuda_transport_host_mem_unregister(const void *guest_va);
/* Returns the registered size, or 0 if not found. */
size_t hmcuda_transport_host_mem_size(const void *guest_va);

/* Sync one kernel arg: if ptr is a registered guest_va, copy guest_va →
 * real_ptr on the backend via the page-pin H2D path.  Clears gpu_dirty.
 * No-op if ptr is not a registered guest_va. */
void hmcuda_transport_sync_host_mem_arg(const void *ptr);

/* Mark a registered guest_va as GPU-dirty: a kernel was just dispatched with
 * this address as an arg and may have written to real_ptr. */
void hmcuda_transport_mark_gpu_dirty(const void *guest_va);

/* Returns true if a kernel has written to real_ptr since the last guest/backend
 * sync.  H2D callers use this to refresh guest_va before taking the page-pin
 * path; pre-launch syncs use it to avoid overwriting backend data. */
bool hmcuda_transport_is_gpu_dirty(const void *guest_va);

/* Clear the gpu_dirty flag for a registered guest_va.  Call after a D2H
 * page-pin copy writes to guest_va so that the next pre-launch sync will
 * push the updated guest_va content to real_ptr. */
void hmcuda_transport_clear_gpu_dirty(const void *guest_va);

#endif /* HMCUDA_TRANSPORT_H */
