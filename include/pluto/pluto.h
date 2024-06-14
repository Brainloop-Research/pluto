// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#ifndef PLUTO_H
#define PLUTO_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration macros

#define PT_ENABLE_LOGGING // Enable logging
#define PT_EXPORT_DLL // Export functions for shared library usage

// Utility macros

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#	define PT_NORET __attribute__((noreturn))
#	define PT_ALIGN(x) __attribute__((aligned(x)))
#	define PT_AINLINE inline __attribute__((always_inline))
#	define PT_NOINLINE __attribute__((noinline))
#   define PT_HOTPROC __attribute__((hot))
#   define PT_COLDPROC __attribute__((cold))
#   define PT_PACKED __attribute__((packed))
#   define PT_FALLTHROUGH __attribute__((fallthrough))
#   define PT_UNUSED __attribute__((unused))
#	define pt_likely(x) __builtin_expect(!!(x), 1)
#	define pt_unlikely(x) __builtin_expect(!!(x), 0)
#else
#	define PT_NORET __declspec(noreturn)
#	define PT_ALIGN(x) __declspec(align(x))
#	define PT_AINLINE inline __forceinline
#	define PT_NOINLINE __declspec(noinline)
#   define PT_HOTPROC
#   define PT_COLDPROC
#   define PT_PACKED __declspec(align(1))
#   define PT_FALLTHROUGH
#   define PT_UNUSED __declspec(unused)
#	define pt_likely(x) (x)
#	define pt_unlikely(x) (x)
#endif

#ifdef PT_EXPORT_DLL
#   ifdef _MSC_VER
#       define PT_EXPORT __declspec(dllexport)
#   else
#       define PT_EXPORT __attribute__((visibility("default")))
#   endif
#else
#   define PT_EXPORT
#endif

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

#define pt_assert_name2(name, line) name ## line
#define pt_assert_name(line) pt_assert_name2(_assert_, line)
#define pt_static_assert(expr) extern void pt_assert_name(__LINE__)(bool STATIC_ASSERTION_FAILED[((expr)?1:-1)])
extern PT_EXPORT PT_NORET PT_COLDPROC void pt_panic(const char *msg, ...); // Panic function

#define pt_assert(expr, msg, ...) \
    if (pt_unlikely(!(expr))) { \
        pt_panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }
#define pt_assert2(expr) pt_assert(expr, "")

#ifdef NDEBUG
#   define pt_dassert(expr, msg, ...)
#   define pt_dassert2(expr)
#else
#   define pt_dassert(expr, msg, ...) pt_assert(expr, msg, ## __VA_ARGS__)
#   define pt_dassert2(expr) pt_assert2(expr)
#endif

#ifdef PT_ENABLE_LOGGING
#   define pt_log_info(msg, ...) fprintf(stdout,  "[pluto] " PT_SRC_NAME " " msg "\n", ## __VA_ARGS__)
#   define pt_log_error(msg, ...) fprintf(stderr,  "[pluto] " PT_SRC_NAME " " PT_CCRED msg PT_CCRESET "\n", ## __VA_ARGS__)
#else
#   define pt_log_info(msg, ...)
#   define pt_log_error(msg, ...)
#endif

typedef void *(*pt_alloc_proc_t)(void *blk, size_t len);
extern PT_EXPORT void *pt_default_allocator(void *blk, size_t len);

#define PT_CTX_CHUNK_SIZE ((size_t)1<<20) // 1 MiB
#define PT_CTX_CHUNKS_CAP 16

struct pt_ctx_t { // Structure to represent a context
    pt_alloc_proc_t alloc; // Allocator function - allows to plug in custom allocators
    size_t chunk_size;     // Size of each chunk
    uint8_t **chunks;    // Allocated chunks
    size_t chunks_len;
    size_t chunks_cap;
    uint8_t *delta;      // Pointer to the next free byte within the current chunk - we allocate downwards
};

extern PT_EXPORT void pt_ctx_init(struct pt_ctx_t *ctx, pt_alloc_proc_t alloc, size_t chunk_size);
extern PT_EXPORT void *pt_ctx_pool_alloc(struct pt_ctx_t *ctx, size_t len);
extern PT_EXPORT void pt_ctx_free(struct pt_ctx_t *ctx);

typedef int64_t pt_dim_t;

#define PT_MAX_DIMS 4

struct pt_tensor_t {        // Structure to represent a tensor
    float *data;                    // Pointer to the data
    pt_dim_t dims[PT_MAX_DIMS];     // Size of each dimension
    pt_dim_t strides[PT_MAX_DIMS];  // Strides for each dimension
    pt_dim_t size;                  // Total size of data in bytes
};

extern PT_EXPORT struct pt_tensor_t *pt_tensor_new(
    const pt_dim_t *dims,
    pt_dim_t num_dims
);

extern PT_EXPORT bool pt_tensor_is_scalar(const struct pt_tensor_t *tensor);
extern PT_EXPORT bool pt_tensor_is_vector(const struct pt_tensor_t *tensor);
extern PT_EXPORT bool pt_tensor_is_matrix(const struct pt_tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif
