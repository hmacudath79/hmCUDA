#ifndef HMCUDA_VHOST_CMD_DISPATCH_H
#define HMCUDA_VHOST_CMD_DISPATCH_H

#include <stdint.h>
#include <stddef.h>
#include "libvhost-user.h"

/*
 * Per-library command dispatch interface for the vhost-user backend.
 *
 * Each library (runtime, driver API, cuBLAS, ...) implements a handle
 * function that processes commands in its ID range.
 *
 * To add a new library:
 *   1. Create cmd_<lib>.c implementing hmcuda_<lib>_handle_cmd()
 *   2. Register it in hmcuda_dispatch_cmd() (see vhost_user_hmcuda.c)
 *   3. Add the .c to the root Makefile source list
 */

struct hmcuda_vhost_cmd_ctx {
    uint32_t session_id;
    uint32_t vm_id;

    /* Request: payload starts after the req_header in the gathered buffer */
    void *payload;
    size_t payload_len;       /* total_out_len - sizeof(req_hdr) */

    /* Response: handler fills these */
    uint8_t *resp_payload;    /* caller-provided buffer */
    size_t resp_payload_cap;  /* capacity of resp_payload buffer */
    size_t resp_payload_len;  /* set by handler */
    uint32_t cuda_error;

    /* Virtqueue element — needed for zero-copy memcpy paths */
    VuVirtqElement *elem;
    VuDev *dev;
    uint8_t *req_buf;         /* full gathered out buffer */
    size_t total_out_len;
    int d2h_virtqueue;        /* set by handler if D2H data written to in_sg */
};

/* CUDA Runtime API */
int hmcuda_runtime_handle_cmd(uint32_t cmd, struct hmcuda_vhost_cmd_ctx *ctx);

/* CUDA Driver API */
int hmcuda_driver_handle_cmd(uint32_t cmd, struct hmcuda_vhost_cmd_ctx *ctx);

/* Future libraries:
 * int hmcuda_cublas_handle_cmd(uint32_t cmd, struct hmcuda_vhost_cmd_ctx *ctx);
 */

#endif /* HMCUDA_VHOST_CMD_DISPATCH_H */
