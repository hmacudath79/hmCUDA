#ifndef VHOST_LOG_H
#define VHOST_LOG_H

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

/* Log levels */
typedef enum {
    LOG_LEVEL_ERROR = 0,
    LOG_LEVEL_WARN  = 1,
    LOG_LEVEL_INFO  = 2,
    LOG_LEVEL_DEBUG = 3,
} LogLevel;

/* Default log level - can be changed at compile time or runtime */
#ifndef VHOST_LOG_LEVEL
#define VHOST_LOG_LEVEL LOG_LEVEL_INFO
#endif

extern LogLevel g_log_level;

/* Get formatted timestamp */
static inline void log_timestamp(char *buf, size_t size) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm *tm = localtime(&tv.tv_sec);
    snprintf(buf, size, "[%02d:%02d:%02d.%03ld]",
             tm->tm_hour, tm->tm_min, tm->tm_sec, tv.tv_usec / 1000);
}

/* Log macros with levels */
#define LOG_ERROR(fmt, ...) do { \
    if (g_log_level >= LOG_LEVEL_ERROR) { \
        char ts[32]; log_timestamp(ts, sizeof(ts)); \
        fprintf(stderr, "%s [ERROR] " fmt "\n", ts, ##__VA_ARGS__); \
        fflush(stderr); \
    } \
} while(0)

#define LOG_WARN(fmt, ...) do { \
    if (g_log_level >= LOG_LEVEL_WARN) { \
        char ts[32]; log_timestamp(ts, sizeof(ts)); \
        fprintf(stderr, "%s [WARN]  " fmt "\n", ts, ##__VA_ARGS__); \
        fflush(stderr); \
    } \
} while(0)

#define LOG_INFO(fmt, ...) do { \
    if (g_log_level >= LOG_LEVEL_INFO) { \
        char ts[32]; log_timestamp(ts, sizeof(ts)); \
        fprintf(stdout, "%s [INFO]  " fmt "\n", ts, ##__VA_ARGS__); \
        fflush(stdout); \
    } \
} while(0)

#define LOG_DEBUG(fmt, ...) do { \
    if (g_log_level >= LOG_LEVEL_DEBUG) { \
        char ts[32]; log_timestamp(ts, sizeof(ts)); \
        fprintf(stdout, "%s [DEBUG] " fmt "\n", ts, ##__VA_ARGS__); \
        fflush(stdout); \
    } \
} while(0)

/* Backward compatibility - maps old LOG() to LOG_INFO() */
#define LOG(fmt, ...) LOG_INFO(fmt, ##__VA_ARGS__)

#endif /* VHOST_LOG_H */
