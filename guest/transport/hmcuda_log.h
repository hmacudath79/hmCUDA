#ifndef HMCUDA_LOG_H
#define HMCUDA_LOG_H

#include <stdio.h>

/*
 * Logging macros for hmCUDA runtime
 * Controlled by HMCUDA_DEBUG environment variable at runtime
 */

/* Log levels */
typedef enum {
    HMCUDA_LOG_NONE   = 0,  /* No logging */
    HMCUDA_LOG_ERROR  = 1,  /* Errors only */
    HMCUDA_LOG_WARN   = 2,  /* Warnings and errors */
    HMCUDA_LOG_INFO   = 3,  /* Info, warnings, and errors */
    HMCUDA_LOG_DEBUG  = 4,  /* All non-timing messages including debug */
    HMCUDA_LOG_TIMING = 5,  /* Per-ioctl latency records only (HMCUDA_DEBUG=5) */
} HmcudaLogLevel;

/* Global log level - set via hmcuda_init() from HMCUDA_DEBUG env var */
extern HmcudaLogLevel g_hmcuda_log_level;

/* Helper to check if logging is enabled at a level.
 * HMCUDA_DEBUG=5 is timing-only, not "debug plus timing". */
#define HMCUDA_LOG_ENABLED(level) \
    ((level) == HMCUDA_LOG_TIMING ? \
        (g_hmcuda_log_level == HMCUDA_LOG_TIMING) : \
        (g_hmcuda_log_level >= (level) && g_hmcuda_log_level < HMCUDA_LOG_TIMING))

/* Log macros */
#define HMCUDA_LOG_ERROR(fmt, ...) do { \
    if (HMCUDA_LOG_ENABLED(HMCUDA_LOG_ERROR)) { \
        fprintf(stderr, "[hmCUDA ERROR] " fmt "\n", ##__VA_ARGS__); \
    } \
} while(0)

#define HMCUDA_LOG_WARN(fmt, ...) do { \
    if (HMCUDA_LOG_ENABLED(HMCUDA_LOG_WARN)) { \
        fprintf(stderr, "[hmCUDA WARN] " fmt "\n", ##__VA_ARGS__); \
    } \
} while(0)

#define HMCUDA_LOG_INFO(fmt, ...) do { \
    if (HMCUDA_LOG_ENABLED(HMCUDA_LOG_INFO)) { \
        fprintf(stdout, "[hmCUDA] " fmt "\n", ##__VA_ARGS__); \
        fflush(stdout); \
    } \
} while(0)

#define HMCUDA_LOG_DEBUG(fmt, ...) do { \
    if (HMCUDA_LOG_ENABLED(HMCUDA_LOG_DEBUG)) { \
        fprintf(stdout, "[hmCUDA DEBUG] " fmt "\n", ##__VA_ARGS__); \
        fflush(stdout); \
    } \
} while(0)

#define HMCUDA_LOG_TIMING(cmd_name, elapsed_us) do { \
    if (HMCUDA_LOG_ENABLED(HMCUDA_LOG_TIMING)) { \
        fprintf(stdout, "[hmCUDA TIMING] cmd=%-40s %lld us\n", \
                (cmd_name), (long long)(elapsed_us)); \
        fflush(stdout); \
    } \
} while(0)

#endif /* HMCUDA_LOG_H */
