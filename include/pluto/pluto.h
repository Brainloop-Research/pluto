// (c) 2024 Brainloop Research

#ifndef PLUTO_H
#define PLUTO_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PT_ENABLE_LOGGING

#define PT_ENUM_SEP ,
#define PT_STRINGIZE(x) PT_STRINGIZE2(x)
#define PT_STRINGIZE2(x) #x
#define PT_CCRED "\x1b[31m"
#define PT_CCGREEN "\x1b[32m"
#define PT_CCYELLOW "\x1b[33m"
#define PT_CCBLUE "\x1b[34m"
#define PT_CCMAGENTA "\x1b[35m"
#define PT_CCCYAN "\x1b[36m"
#define PT_CCRESET "\x1b[0m"

#define PT_SRC_NAME __FILE_NAME__ ":" PT_STRINGIZE(__LINE__)

#ifdef PT_ENABLE_LOGGING
#   define pt_info(msg, ...) fprintf(stdout,  "[pluto] " PT_SRC_NAME " " msg "\n", ## __VA_ARGS__)
#   define pt_warn(msg, ...) fprintf(stderr,  "[pluto] " PT_SRC_NAME " " PT_CCRED msg PT_CCRESET "\n", ## __VA_ARGS__)
#else
#   define pt_info(msg, ...)
#   define pt_warn(msg, ...)
#endif

typedef void *(*pt_alloc_proc_t)(void *blk, size_t len);
extern void *pt_default_allocator(void *blk, size_t len);

#define PT_CTX_CHUNK_SIZE ((size_t)1<<20) // 1 MB
#define PT_CTX_CHUNKS_CAP 16

typedef struct pt_ctx_t { // Structure to represent a context
    pt_alloc_proc_t alloc; // Allocator function
    size_t chunk_size;     // Size of each chunk
    uint8_t **chunks;    // Allocated chunks
    size_t chunks_len;
    size_t chunks_cap;
    uint8_t *delta;
} pt_ctx_t;

extern void pt_ctx_init(pt_ctx_t *ctx, pt_alloc_proc_t alloc, size_t chunk_size);
extern void *pt_ctx_pool_alloc(pt_ctx_t *ctx, size_t len);
extern void pt_ctx_free(pt_ctx_t *ctx);

typedef int64_t pt_dim_t;

#define PT_MAX_DIMS 4

typedef struct pt_tensor_t {        // Structure to represent a tensor
    float *data;                    // Pointer to the data
    pt_dim_t dims[PT_MAX_DIMS];     // Size of each dimension
    pt_dim_t strides[PT_MAX_DIMS];  // Strides for each dimension
    pt_dim_t size;                  // Total size of data in bytes
} pt_tensor_t;

extern pt_tensor_t *pt_tensor_new(
    const pt_dim_t *dims,
    pt_dim_t num_dims
);

#ifdef __cplusplus
}
#endif

#endif
