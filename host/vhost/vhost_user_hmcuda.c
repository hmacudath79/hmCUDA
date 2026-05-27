#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <signal.h>
#include <glib.h>
#include <stddef.h>
#include <time.h>
#include <sys/time.h>

#include "libvhost-user-glib.h"
#include "hmcuda_api.h"
#include "cmd_dispatch.h"
#include "log.h"

/* Extern reference to log level defined in core.c */
extern LogLevel g_log_level;

/* Global VuDev pointer, set after vug_init so core_driver.c can access
 * guest memory regions for cuMemHostRegister. */
static VuDev *g_vu_dev = NULL;

VuDev *hmcuda_vu_dev_get(void) { return g_vu_dev; }

typedef struct VhostUserHmCuda {
    VugDev dev;
    int session_id;
    int vm_id;
} VhostUserHmCuda;

#ifndef container_of
#define container_of(ptr, type, member) ({ \
    const typeof(((type *)0)->member) *__mptr = (ptr); \
    (type *)((char *)__mptr - offsetof(type, member)); })
#endif

static size_t iov_to_buf(const struct iovec *iov, unsigned int iov_cnt, void *buf, size_t len)
{
    size_t offset = 0;
    unsigned int i;
    for (i = 0; i < iov_cnt && offset < len; i++) {
        size_t copy = len - offset;
        if (copy > iov[i].iov_len)
            copy = iov[i].iov_len;
        memcpy((char *)buf + offset, iov[i].iov_base, copy);
        offset += copy;
    }
    return offset;
}

static size_t buf_to_iov(const struct iovec *iov, unsigned int iov_cnt, const void *buf, size_t len)
{
    size_t offset = 0;
    unsigned int i;
    for (i = 0; i < iov_cnt && offset < len; i++) {
        size_t copy = len - offset;
        if (copy > iov[i].iov_len)
            copy = iov[i].iov_len;
        memcpy(iov[i].iov_base, (char *)buf + offset, copy);
        offset += copy;
    }
    return offset;
}

/*
 * Dispatch a command to the appropriate library handler.
 * Returns 0 if handled, -1 if unknown.
 */
static int hmcuda_dispatch_cmd(uint32_t cmd, struct hmcuda_vhost_cmd_ctx *ctx)
{
    int ret;

    ret = hmcuda_runtime_handle_cmd(cmd, ctx);
    if (ret) ret = hmcuda_driver_handle_cmd(cmd, ctx);
    /* Future libraries:
     * if (ret) ret = hmcuda_cublas_handle_cmd(cmd, ctx);
     */

    return ret;
}

static void hmcuda_process_vq(VuDev *dev, int qidx)
{
    VuVirtq *vq = vu_get_queue(dev, qidx);
    VuVirtqElement *elem;
    VugDev *gdev = container_of(dev, VugDev, parent);
    VhostUserHmCuda *hdev = container_of(gdev, VhostUserHmCuda, dev);
    uint32_t sid = hdev->session_id;
    uint32_t vid = hdev->vm_id;

    while ((elem = vu_queue_pop(dev, vq, sizeof(VuVirtqElement))) != NULL) {
        struct hmcuda_req_header req_hdr;
        struct hmcuda_resp_header resp_hdr = {0};
        size_t read_len;
        
        read_len = iov_to_buf(elem->out_sg, elem->out_num, &req_hdr, sizeof(req_hdr));
        if (read_len != sizeof(req_hdr)) {
            LOG("Short read on request header");
            free(elem);
            continue;
        }

        resp_hdr.cmd_type = req_hdr.cmd_type;
        
        size_t total_out_len = 0;
        for (unsigned i = 0; i < elem->out_num; i++) {
            total_out_len += elem->out_sg[i].iov_len;
        }
        
        /* 
         * Zero-copy optimization: Only extract the request struct into req_buf.
         * elem->out_sg[1..N] contains the bulk data pages. We leave them in place 
         * for handlers to DMA directly from iov_base, avoiding a massive CPU copy.
         */
        size_t req_struct_len = elem->out_sg[0].iov_len;
        uint8_t *req_buf = malloc(req_struct_len);
        memcpy(req_buf, elem->out_sg[0].iov_base, req_struct_len);
        
        void *payload = req_buf + sizeof(req_hdr);
        
        uint8_t resp_payload[2048];
        size_t resp_payload_len = 0;
        int d2h_virtqueue = 0;

        struct hmcuda_vhost_cmd_ctx cmd_ctx = {
            .session_id = sid,
            .vm_id = vid,
            .payload = payload,
            .payload_len = req_struct_len - sizeof(req_hdr),
            .resp_payload = resp_payload,
            .resp_payload_cap = sizeof(resp_payload),
            .resp_payload_len = 0,
            .cuda_error = 0,
            .elem = elem,
            .dev = dev,
            .req_buf = req_buf,
            .total_out_len = total_out_len,
            .d2h_virtqueue = 0,
        };

        if (hmcuda_dispatch_cmd(req_hdr.cmd_type, &cmd_ctx) != 0) {
            cmd_ctx.cuda_error = 999;
        }

        resp_hdr.cuda_error = cmd_ctx.cuda_error;
        resp_payload_len = cmd_ctx.resp_payload_len;
        d2h_virtqueue = cmd_ctx.d2h_virtqueue;

        free(req_buf);

        size_t total_resp_len = sizeof(resp_hdr) + resp_payload_len;
        uint8_t *resp_buf = malloc(total_resp_len);
        memcpy(resp_buf, &resp_hdr, sizeof(resp_hdr));
        if (resp_payload_len > 0) {
            memcpy(resp_buf + sizeof(resp_hdr), resp_payload, resp_payload_len);
        }

        if (d2h_virtqueue && elem->in_num >= 2) {
            /* D2H with virtqueue data: in_sg[0]=data (already filled), in_sg[1]=response.
             * Write response to the LAST in_sg entry to avoid overwriting GPU data. */
            struct iovec *resp_iov = &elem->in_sg[elem->in_num - 1];
            size_t copy = total_resp_len < resp_iov->iov_len ? total_resp_len : resp_iov->iov_len;
            memcpy(resp_iov->iov_base, resp_buf, copy);
            LOG_DEBUG("[MEMCPY] D2H: Response written to in_sg[%u] (%zu bytes)", elem->in_num - 1, copy);
        } else {
            buf_to_iov(elem->in_sg, elem->in_num, resp_buf, total_resp_len);
        }
        free(resp_buf);

        /* For D2H virtqueue, total bytes written = data pages + response */
        size_t push_len = total_resp_len;
        if (d2h_virtqueue && elem->in_num >= 2) {
            for (unsigned i = 0; i < elem->in_num - 1; i++)
                push_len += elem->in_sg[i].iov_len;
        }
        vu_queue_push(dev, vq, elem, push_len);
        vu_queue_notify(dev, vq);
        free(elem);
    }
}

static void panic_cb(VuDev *dev, const char *msg)
{
    LOG("PANIC: %s", msg);
    exit(EXIT_FAILURE);
}

static uint64_t hmcuda_get_features(VuDev *dev)
{
    LOG("hmcuda_get_features");
    return 0;
}

static void hmcuda_set_features(VuDev *dev, uint64_t features)
{
    LOG("hmcuda_set_features: 0x%lx", features);
}

static int hmcuda_process_msg(VuDev *dev, VhostUserMsg *vmsg, int *do_reply)
{
    switch (vmsg->request) {
    case VHOST_USER_SET_MEM_TABLE:
        LOG("hmcuda_process_msg: VHOST_USER_SET_MEM_TABLE");
        break;
    case VHOST_USER_SET_VRING_ADDR:
        LOG("hmcuda_process_msg: VHOST_USER_SET_VRING_ADDR");
        break;
    default:
        break;
    }
    return 0;
}

static const VuDevIface iface = {
    .get_features = hmcuda_get_features,
    .set_features = hmcuda_set_features,
    .process_msg = hmcuda_process_msg,
};

int main(int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    VhostUserHmCuda hdev = {0};
    int socket_fd;
    char *socket_path;

    if (argc != 2) {
        LOG("Usage: %s <socket_path>", argv[0]);
        LOG("  Set HMCUDA_LOG_LEVEL=0..3 to control verbosity (0=error, 1=warn, 2=info, 3=debug)");
        return -1;
    }
    socket_path = argv[1];

    const char *log_env = getenv("HMCUDA_LOG_LEVEL");
    if (log_env) {
        int level = atoi(log_env);
        if (level >= LOG_LEVEL_ERROR && level <= LOG_LEVEL_DEBUG)
            g_log_level = (LogLevel)level;
    }

    loop = g_main_loop_new(NULL, FALSE);

    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        perror("socket");
        return -1;
    }

    struct sockaddr_un un;
    un.sun_family = AF_UNIX;
    strncpy(un.sun_path, socket_path, sizeof(un.sun_path) - 1);
    
    unlink(socket_path);
    if (bind(socket_fd, (struct sockaddr *)&un, sizeof(un)) < 0) {
        perror("bind");
        close(socket_fd);
        return -1;
    }

    if (listen(socket_fd, 1) < 0) {
        perror("listen");
        close(socket_fd);
        return -1;
    }

    LOG("Waiting for connection on %s...", socket_path);
    int conn_fd = accept(socket_fd, NULL, NULL);
    if (conn_fd < 0) {
        perror("accept");
        close(socket_fd);
        return -1;
    }

    hdev.session_id = 1;
    hdev.vm_id = 1;

    if (!vug_init(&hdev.dev, 2, conn_fd, panic_cb, &iface)) {
        LOG("Failed to initialize vhost-user device\n");
        close(conn_fd);
        close(socket_fd);
        return -1;
    }
    g_vu_dev = &hdev.dev.parent;

    vu_set_queue_handler(&hdev.dev.parent, &hdev.dev.parent.vq[0], hmcuda_process_vq);

    LOG("vhost-user-hmcuda started, waiting for requests...");
    g_main_loop_run(loop);

    vug_deinit(&hdev.dev);
    g_main_loop_unref(loop);
    close(conn_fd);
    close(socket_fd);
    unlink(socket_path);

    return 0;
}
