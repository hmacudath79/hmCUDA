#ifndef HMCUDA_DISPATCH_H
#define HMCUDA_DISPATCH_H

/*
 * Dispatch macros for intercept libraries.
 *
 * These eliminate boilerplate in the ~40 intercepted API functions that
 * follow the same dispatch pattern.  Each macro takes a dispatch function
 * (hmcuda_dispatch / hmcuda_drv_dispatch) so both runtime and driver
 * intercept libs can share them.
 */

/* Dispatch request, no response.  Returns the dispatch result directly. */
#define HMCUDA_FORWARD(fn, cmd, req) \
    return fn(cmd, &(req), sizeof(req), nullptr, 0)

/* Dispatch with no request and no response. */
#define HMCUDA_FORWARD_NOREQ(fn, cmd) \
    return fn(cmd, nullptr, 0, nullptr, 0)

/* Dispatch request, write resp.<field> → *out_ptr on success. */
#define HMCUDA_FORWARD_RESP(fn, cmd, req, resp_t, out_ptr, cast, field) \
    do { \
        resp_t _resp; \
        auto _err = fn(cmd, &(req), sizeof(req), &_resp, sizeof(_resp)); \
        if (!_err && (out_ptr)) *(out_ptr) = cast _resp.field; \
        return _err; \
    } while(0)

/* Dispatch with no request, write resp.<field> → *out_ptr on success. */
#define HMCUDA_FORWARD_NOREQ_RESP(fn, cmd, resp_t, out_ptr, cast, field) \
    do { \
        resp_t _resp; \
        auto _err = fn(cmd, nullptr, 0, &_resp, sizeof(_resp)); \
        if (!_err && (out_ptr)) *(out_ptr) = cast _resp.field; \
        return _err; \
    } while(0)

#endif /* HMCUDA_DISPATCH_H */
