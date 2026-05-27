// SPDX-License-Identifier: GPL-2.0-only
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/virtio.h>
#include <linux/virtio_ids.h>
#include <linux/virtio_config.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/wait.h>
#include <linux/sched.h>
#include <linux/scatterlist.h>
#include <linux/io.h>
#include <linux/dma-mapping.h>
#include <linux/mm.h>

#include "hmcuda_api.h"
#include "cmd_dispatch.h"

#define DRIVER_NAME "virtio_hmcuda"
#define DEVICE_NAME "virtio-hmcuda"

// Arbitrary VirtIO Device ID for hmCUDA (Must match QEMU)
#define VIRTIO_ID_HMCUDA 30

#define HM_CUDA_IOC_MAGIC 'k'
#define HM_CUDA_IOC_REQ _IOWR(HM_CUDA_IOC_MAGIC, 1, struct hmcuda_req_header)

#define HMCUDA_MEMCPY_PAGES_PER_BATCH 512

/* 0 = silent, 1 = errors only, 2 = info+errors (default) */
static int hmcuda_log_level = 2;
module_param(hmcuda_log_level, int, 0644);
MODULE_PARM_DESC(hmcuda_log_level, "Log verbosity: 0=off 1=errors 2=info+errors");

#define hmcuda_info(dev, fmt, ...)  do { if (hmcuda_log_level >= 2) dev_info(dev, fmt, ##__VA_ARGS__); } while (0)
#define hmcuda_err(dev, fmt, ...)   do { if (hmcuda_log_level >= 1) dev_err(dev,  fmt, ##__VA_ARGS__); } while (0)

struct virtio_hmcuda_device {
    struct virtio_device *vdev;
    struct virtqueue *vq;
    struct miscdevice misc;
    struct mutex lock;
    struct completion completion;
};

/*
 * Dispatch command to the appropriate library's size-lookup function.
 * Tries each library in turn until one claims the command.
 * Add new libraries here as they are implemented.
 */
static int hmcuda_cmd_get_sizes(uint32_t cmd, const void *req_payload,
                                struct hmcuda_cmd_sizes *sizes)
{
    int ret;

    ret = hmcuda_runtime_get_sizes(cmd, req_payload, sizes);
    if (ret) ret = hmcuda_driver_get_sizes(cmd, req_payload, sizes);
    /* Future libraries:
     * if (ret) ret = hmcuda_cublas_get_sizes(cmd, req_payload, sizes);
     */

    return ret;
}

static void hmcuda_vq_callback(struct virtqueue *vq)
{
    struct virtio_hmcuda_device *hdev = vq->vdev->priv;
    complete(&hdev->completion);
}

/* ------------------------------------------------------------------ */
/*  Helper: two-pass size discovery                                    */
/*  Pass 1: fixed sizes with NULL payload.                            */
/*  Pass 2: re-query with the real payload to account for variable-   */
/*  length trailing data (e.g. args_size, device_name_len).           */
/* ------------------------------------------------------------------ */
static int hmcuda_get_req_sizes(uint32_t cmd, unsigned long arg,
                                size_t *req_len, size_t *resp_len)
{
    struct hmcuda_cmd_sizes sizes;
    int ret;

    ret = hmcuda_cmd_get_sizes(cmd, NULL, &sizes);
    if (ret)
        return ret;
    *req_len  = sizes.req_size;
    *resp_len = sizes.resp_size;

    if (*req_len > 0) {
        void *fixed_req = kmalloc(*req_len, GFP_KERNEL);
        if (!fixed_req)
            return -ENOMEM;
        if (copy_from_user(fixed_req,
                           (void __user *)(arg + sizeof(struct hmcuda_req_header)),
                           *req_len)) {
            kfree(fixed_req);
            return -EFAULT;
        }
        ret = hmcuda_cmd_get_sizes(cmd, fixed_req, &sizes);
        kfree(fixed_req);
        if (ret)
            return ret;
        *req_len = sizes.req_size;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Helper: allocate and populate req/resp kernel buffers              */
/* ------------------------------------------------------------------ */
static int hmcuda_build_req_buf(struct hmcuda_req_header *hdr, unsigned long arg,
                                size_t req_len, size_t resp_len,
                                void **req_buf_out, void **resp_buf_out,
                                size_t *total_req_out, size_t *total_resp_out)
{
    size_t total_req  = sizeof(*hdr) + req_len;
    size_t total_resp = sizeof(struct hmcuda_resp_header) + resp_len;
    void  *req_buf, *resp_buf;

    req_buf = kzalloc(total_req, GFP_KERNEL);
    if (!req_buf)
        return -ENOMEM;

    resp_buf = kzalloc(total_resp, GFP_KERNEL);
    if (!resp_buf) {
        kfree(req_buf);
        return -ENOMEM;
    }

    memcpy(req_buf, hdr, sizeof(*hdr));
    if (req_len > 0 &&
        copy_from_user(req_buf + sizeof(*hdr),
                       (void __user *)(arg + sizeof(*hdr)), req_len)) {
        kfree(resp_buf);
        kfree(req_buf);
        return -EFAULT;
    }

    *req_buf_out    = req_buf;
    *resp_buf_out   = resp_buf;
    *total_req_out  = total_req;
    *total_resp_out = total_resp;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Memcpy-family command args                                         */
/* ------------------------------------------------------------------ */
struct hmcuda_memcpy_args {
    uint64_t     user_addr_start;
    uint64_t     total_count;
    uint64_t     dev_ptr;
    unsigned int kind;          /* 1 = H2D, 2 = D2H */
};

static void hmcuda_parse_memcpy_args(uint32_t cmd, void *req_payload,
                                     struct hmcuda_memcpy_args *out)
{
    if (cmd == HMCUDA_CMD_CU_MEM_HOST_ALLOC) {
        struct hmcuda_cu_mem_host_alloc_req *r = req_payload;
        out->user_addr_start = r->guest_va;
        out->total_count     = r->bytesize;
        out->dev_ptr         = 0;
        out->kind            = 1; /* always H2D */
    } else {
        struct hmcuda_memcpy_req *r = req_payload;
        out->total_count     = r->count;
        out->kind            = r->kind;
        out->user_addr_start = (r->kind == 1) ? r->src : r->dst;
        out->dev_ptr         = (r->kind == 1) ? r->dst : r->src;
    }
}

/* ------------------------------------------------------------------ */
/*  GUP scatter-gather memcpy loop                                     */
/*                                                                     */
/*  User pages are pinned via GUP and passed directly as vring        */
/*  descriptors.  The host vhost reads/writes through its guest       */
/*  memory mapping — no copy_from/to_user involved.                   */
/*                                                                     */
/*  Physically contiguous page runs are coalesced into single SG      */
/*  entries to minimise vring descriptor count.                       */
/*                                                                     */
/*  Returns 0 on success or CUDA error (resp_buf populated);          */
/*  negative errno on kernel error (resp_buf not valid).              */
/* ------------------------------------------------------------------ */
static int hmcuda_do_memcpy_gup(struct virtio_hmcuda_device *hdev,
                                void *req_buf, size_t total_req_len,
                                void *resp_buf, size_t total_resp_len,
                                const struct hmcuda_memcpy_args *args)
{
    uint32_t cmd = ((struct hmcuda_req_header *)req_buf)->cmd_type;
    unsigned long first_page_off = args->user_addr_start & ~PAGE_MASK;
    long total_nr_pages  = DIV_ROUND_UP(first_page_off + args->total_count, PAGE_SIZE);
    unsigned int gup_flags = (args->kind == 2) ? FOLL_WRITE : 0;
    int max_pages_per_batch = HMCUDA_MEMCPY_PAGES_PER_BATCH;
    struct page      **pages;
    struct scatterlist *sg_batch;
    struct scatterlist  sg[2];
    uint64_t byte_offset = 0;
    int      batch_num   = 0;
    int      ret = 0;

    hmcuda_info(&hdev->vdev->dev,
                "[MEMCPY] kind=%u, total=%llu bytes, total_pages=%ld, max_pages/batch=%d\n",
                args->kind, args->total_count, total_nr_pages, max_pages_per_batch);

    pages = kvmalloc_array(max_pages_per_batch, sizeof(*pages), GFP_KERNEL);
    if (!pages)
        return -ENOMEM;

    /* worst-case: no contiguous runs, one SG entry per page */
    sg_batch = kvmalloc_array(max_pages_per_batch, sizeof(*sg_batch), GFP_KERNEL);
    if (!sg_batch) {
        kvfree(pages);
        return -ENOMEM;
    }

    while (byte_offset < args->total_count) {
        uint64_t user_addr       = args->user_addr_start + byte_offset;
        unsigned long cur_page_off = user_addr & ~PAGE_MASK;
        uint64_t remaining       = args->total_count - byte_offset;
        long pages_remaining     = DIV_ROUND_UP(cur_page_off + remaining, PAGE_SIZE);
        int  pages_this_batch    = (int)min_t(long, max_pages_per_batch, pages_remaining);
        uint64_t capacity        = (uint64_t)pages_this_batch * PAGE_SIZE - cur_page_off;
        uint64_t current_count   = min(capacity, remaining);
        struct scatterlist *sgs_memcpy[3] = { &sg[0], sg_batch, &sg[1] };
        long pinned;
        int  nr_sge, i;
        size_t rem;
        unsigned int len;
        batch_num++;

        pinned = pin_user_pages_fast(user_addr & PAGE_MASK,
                                     pages_this_batch, gup_flags, pages);
        if (pinned != pages_this_batch) {
            if (pinned > 0)
                unpin_user_pages(pages, pinned);
            ret = (pinned < 0) ? (int)pinned : -EFAULT;
            goto out;
        }

        /* Greedy SG coalescing: extend current entry while PFNs are contiguous. */
        sg_init_table(sg_batch, pages_this_batch);
        rem    = (size_t)current_count;
        nr_sge = 0;
        for (i = 0; i < pages_this_batch && rem > 0; i++) {
            size_t pg_off = (i == 0) ? cur_page_off : 0;
            size_t pg_len = min_t(size_t, PAGE_SIZE - pg_off, rem);
            if (nr_sge > 0 &&
                page_to_pfn(pages[i]) == page_to_pfn(pages[i - 1]) + 1) {
                sg_batch[nr_sge - 1].length += pg_len;
            } else {
                sg_set_page(&sg_batch[nr_sge], pages[i], pg_len, pg_off);
                nr_sge++;
            }
            rem -= pg_len;
        }
        sg_mark_end(&sg_batch[nr_sge - 1]);

        if (batch_num == 1 || batch_num % 1000 == 0)
            hmcuda_info(&hdev->vdev->dev,
                        "[MEMCPY] Batch %d: byte_off=%llu, size=%llu, pages=%d, sge=%d\n",
                        batch_num, byte_offset, current_count, pages_this_batch, nr_sge);

        /* Update per-batch fields in the request struct before submission. */
        if (cmd == HMCUDA_CMD_CU_MEM_HOST_ALLOC) {
            struct hmcuda_cu_mem_host_alloc_req *r =
                req_buf + sizeof(struct hmcuda_req_header);
            r->byte_offset = byte_offset;
        } else {
            struct hmcuda_memcpy_req *r =
                req_buf + sizeof(struct hmcuda_req_header);
            r->count = current_count;
            if (args->kind == 1) { /* H2D */
                r->src = 0;
                r->dst = args->dev_ptr + byte_offset;
            } else {               /* D2H */
                r->src = args->dev_ptr + byte_offset;
                /* Destination data is carried by the writable virtqueue SGs.
                 * Keep dst zero so the vhost backend uses the D2H virtqueue
                 * path instead of trying to translate a guest virtual address
                 * as a legacy GPA. */
                r->dst = 0;
            }
        }

        mutex_lock(&hdev->lock);
        reinit_completion(&hdev->completion);

        sg_init_one(&sg[0], req_buf, total_req_len);
        sg_init_one(&sg[1], resp_buf, total_resp_len);

        /* H2D: out=[req, data], in=[resp] — D2H: out=[req], in=[data, resp] */
        if (args->kind == 1)
            ret = virtqueue_add_sgs(hdev->vq, sgs_memcpy, 2, 1, req_buf, GFP_KERNEL);
        else
            ret = virtqueue_add_sgs(hdev->vq, sgs_memcpy, 1, 2, req_buf, GFP_KERNEL);

        if (ret < 0) {
            hmcuda_err(&hdev->vdev->dev, "[MEMCPY] virtqueue_add_sgs failed: %d\n", ret);
            mutex_unlock(&hdev->lock);
            unpin_user_pages(pages, pages_this_batch);
            goto out;
        }

        virtqueue_kick(hdev->vq);
        wait_for_completion(&hdev->completion);

        virtqueue_get_buf(hdev->vq, &len);
        if (batch_num == 1 || batch_num % 1000 == 0)
            hmcuda_info(&hdev->vdev->dev, "[MEMCPY] Batch %d: response len=%u\n",
                        batch_num, len);
        mutex_unlock(&hdev->lock);

        /* Host is done with the pages; safe to unpin. */
        unpin_user_pages(pages, pages_this_batch);

        {
            struct hmcuda_resp_header *resp_hdr = resp_buf;
            if (resp_hdr->cuda_error) {
                hmcuda_err(&hdev->vdev->dev, "[MEMCPY] Batch %d: CUDA error=%u\n",
                           batch_num, resp_hdr->cuda_error);
                goto out; /* resp_buf populated; caller will copy_to_user */
            }
        }

        byte_offset += current_count;
    }

    hmcuda_info(&hdev->vdev->dev, "[MEMCPY] Completed in %d batch(es)\n", batch_num);
out:
    kvfree(sg_batch);
    kvfree(pages);
    return ret;
}

/* ------------------------------------------------------------------ */
/*  Simple (non-memcpy) command dispatch                               */
/* ------------------------------------------------------------------ */
static int hmcuda_do_simple_cmd(struct virtio_hmcuda_device *hdev,
                                void *req_buf, size_t total_req_len,
                                void *resp_buf, size_t total_resp_len)
{
    struct scatterlist  sg[2];
    struct scatterlist *sgs[2] = { &sg[0], &sg[1] };
    unsigned int len;
    int ret;

    mutex_lock(&hdev->lock);
    reinit_completion(&hdev->completion);

    sg_init_one(&sg[0], req_buf, total_req_len);
    sg_init_one(&sg[1], resp_buf, total_resp_len);

    ret = virtqueue_add_sgs(hdev->vq, sgs, 1, 1, req_buf, GFP_KERNEL);
    if (ret < 0) {
        mutex_unlock(&hdev->lock);
        return ret;
    }

    virtqueue_kick(hdev->vq);
    wait_for_completion(&hdev->completion);

    virtqueue_get_buf(hdev->vq, &len);
    mutex_unlock(&hdev->lock);

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Main ioctl dispatcher                                              */
/* ------------------------------------------------------------------ */
static long hmcuda_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct virtio_hmcuda_device *hdev = filp->private_data;
    struct hmcuda_req_header header;
    void   *req_buf = NULL, *resp_buf = NULL;
    size_t  req_len = 0, resp_len = 0;
    size_t  total_req_len, total_resp_len;
    int     ret;

    if (cmd != HM_CUDA_IOC_REQ)
        return -EINVAL;

    if (copy_from_user(&header, (void __user *)arg, sizeof(header)))
        return -EFAULT;

    ret = hmcuda_get_req_sizes(header.cmd_type, arg, &req_len, &resp_len);
    if (ret)
        return ret;

    ret = hmcuda_build_req_buf(&header, arg, req_len, resp_len,
                               &req_buf, &resp_buf, &total_req_len, &total_resp_len);
    if (ret)
        return ret;

    if (header.cmd_type == HMCUDA_CMD_MEMCPY ||
        header.cmd_type == HMCUDA_CMD_CU_MEMCPY ||
        header.cmd_type == HMCUDA_CMD_CU_MEM_HOST_ALLOC) {
        struct hmcuda_memcpy_args args;
        hmcuda_parse_memcpy_args(header.cmd_type, req_buf + sizeof(header), &args);
        ret = hmcuda_do_memcpy_gup(hdev, req_buf, total_req_len,
                                   resp_buf, total_resp_len, &args);
    } else {
        ret = hmcuda_do_simple_cmd(hdev, req_buf, total_req_len,
                                   resp_buf, total_resp_len);
    }

    if (ret == 0 && copy_to_user((void __user *)arg, resp_buf, total_resp_len))
        ret = -EFAULT;

    kfree(req_buf);
    kfree(resp_buf);
    return ret;
}

static int hmcuda_open(struct inode *inode, struct file *filp)
{
    struct virtio_hmcuda_device *hdev = container_of(filp->private_data, struct virtio_hmcuda_device, misc);
    filp->private_data = hdev;
    return 0;
}

static int hmcuda_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static const struct file_operations hmcuda_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = hmcuda_ioctl,
    .open = hmcuda_open,
    .release = hmcuda_release,
};

static int virtio_hmcuda_probe(struct virtio_device *vdev)
{
    struct virtio_hmcuda_device *hdev;
    int ret;

    hdev = kzalloc(sizeof(*hdev), GFP_KERNEL);
    if (!hdev)
        return -ENOMEM;

    hdev->vdev = vdev;
    vdev->priv = hdev;
    mutex_init(&hdev->lock);
    init_completion(&hdev->completion);

    hdev->vq = virtio_find_single_vq(vdev, hmcuda_vq_callback, "requests");
    if (IS_ERR(hdev->vq)) {
        ret = PTR_ERR(hdev->vq);
        goto err_free;
    }
    hmcuda_info(&vdev->dev, "vring size = %u\n", virtqueue_get_vring_size(hdev->vq));

    hdev->misc.minor = MISC_DYNAMIC_MINOR;
    hdev->misc.name = DEVICE_NAME;
    hdev->misc.fops = &hmcuda_fops;
    hdev->misc.parent = &vdev->dev;

    ret = misc_register(&hdev->misc);
    if (ret)
        goto err_del_vq;

    virtio_device_ready(vdev);
    return 0;

err_del_vq:
    vdev->config->del_vqs(vdev);
err_free:
    kfree(hdev);
    return ret;
}

static void virtio_hmcuda_remove(struct virtio_device *vdev)
{
    struct virtio_hmcuda_device *hdev = vdev->priv;

    misc_deregister(&hdev->misc);
    vdev->config->reset(vdev);
    vdev->config->del_vqs(vdev);
    kfree(hdev);
}

static const struct virtio_device_id id_table[] = {
    { VIRTIO_ID_HMCUDA, VIRTIO_DEV_ANY_ID },
    { 0 },
};
MODULE_DEVICE_TABLE(virtio, id_table);

static struct virtio_driver virtio_hmcuda_driver = {
    .driver.name = DRIVER_NAME,
    .driver.owner = THIS_MODULE,
    .id_table = id_table,
    .probe = virtio_hmcuda_probe,
    .remove = virtio_hmcuda_remove,
};

module_virtio_driver(virtio_hmcuda_driver);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("VirtIO hmCUDA Driver");
MODULE_AUTHOR("hmCUDA Team");
