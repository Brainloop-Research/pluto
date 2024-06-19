// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "pluto.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#ifdef __APPLE__
#   include <sys/sysctl.h>
#   include <sys/types.h>
#elif defined(__x86_64__) && !defined(_WIN32)
#   include <cpuid.h>
#elif defined(_MSC_VER)
#   include <intrin.h>
#endif

void pt_panic(const char *const msg, ...) {
    fprintf(stderr, "%s", PT_CCRED);
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    fprintf(stderr, "%s", PT_CCRESET);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

void *pt_default_allocator(void *blk, const size_t len) {
    if (!len) {
        free(blk);
        return NULL;
    } else if(!blk) {
        blk = malloc(len);
        pt_assert(blk, "Failed to allocate %.03fKiB memory", (double)len/(double)(1<<10));
        return blk;
    } else {
        void *const block = realloc(blk, len);
        pt_assert(blk, "Failed to reallocate %.03fKiB memory", (double)len/(double)(1<<10));
        return block;
    }
}

static void pt_ctx_push_chunk(struct pt_ctx_t *const ctx) {
    uint8_t *const chunk = (*ctx->alloc)(NULL, ctx->chunk_size);
    ctx->mapped_total += ctx->chunk_size;
    if (ctx->chunks_len == ctx->chunks_cap)
        ctx->chunks = (*ctx->alloc)(ctx->chunks, (ctx->chunks_cap <<= 1) * sizeof(*ctx->chunks));
    ctx->chunks[ctx->chunks_len++] = chunk;
    ctx->delta = chunk + ctx->chunk_size;
}

#if !defined(__ARM_ARCH) && (defined(__x86_64__) || defined(_WIN64))
static inline void pt_cpuid(uint32_t (*const o)[4], const uint32_t x) {
#ifdef _MSC_VER
    __cpuidex(out, x, 0);
#elif defined(__cpuid_count)
    __cpuid_count(x, 0, (*o)[0], (*o)[1], (*o)[2], (*o)[3]);
#else
    __asm__ __volatile__(
        "xchgq  %%rbx,%q1\n"
        "cpuid\n"
        "xchgq  %%rbx,%q1"
        : "=a"((*o)[0]), "=r" ((*o)[1]), "=c"((*o)[2]), "=d"((*o)[3])
        : "0"(x), "2"(0)
    );
#endif
}
#endif

static void pt_query_cpu_name(struct pt_ctx_t *const ctx) {
#if defined(__APPLE__) && defined(__aarch64__)
    const char* const id = "machdep.cpu.brand_string";
    size_t len;
    if (pt_unlikely(sysctlbyname(id, NULL, &len, NULL, 0) != 0)) return;
    char *const scratch = alloca(len+1);
    if (pt_unlikely(sysctlbyname(id, scratch, &len, NULL, 0) != 0)) return;
    scratch[len] = '\0';
    strncpy(ctx->cpu_name, scratch, sizeof(ctx->cpu_name));
#elif !defined(__ARM_ARCH) && (defined(__x86_64__) || defined(_WIN64))
    uint32_t regs[4];
    pt_assert2(sizeof(ctx->cpu_name) >= sizeof(regs));
    pt_cpuid(&regs, 0);
    char *const name = ctx->cpu_name;
    pt_cpuid(&regs, 0x80000002);
    for (int i = 0; i < 4; ++i)
        *(uint32_t *)(name+(i<<2)) = regs[i];
    pt_cpuid(&regs, 0x80000003);
    for (int i = 0; i < 4; ++i)
        *(uint32_t *)(name+(i<<2)+16) = regs[i];
    pt_cpuid(&regs, 0x80000004);
    for (int i = 0; i < 4; ++i)
        *(uint32_t *)(name+(i<<2)+32) = regs[i];
#else
    strcpy(ctx->cpu_name, "Unknown");
#endif
}

void pt_ctx_init(struct pt_ctx_t *const ctx, const pt_alloc_proc_t alloc, const size_t chunk_size) {
    if (chunk_size > 1 && chunk_size < (1<<20))
        pt_log_error("Chunk size very small, set it to >= 1MiB for best performance");
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc = alloc ? alloc : &pt_default_allocator;
    ctx->chunk_size = chunk_size ? chunk_size : PT_CTX_CHUNK_SIZE;
    ctx->chunks_cap = PT_CTX_CHUNKS_CAP;
    ctx->chunks = (*ctx->alloc)(NULL, ctx->chunks_cap * sizeof(*ctx->chunks));
    pt_ctx_push_chunk(ctx);
    pt_query_cpu_name(ctx);
    printf("CPU: %s\n", ctx->cpu_name);
}

void *pt_ctx_pool_alloc(struct pt_ctx_t *const ctx, const size_t len) {
    pt_assert(len && len <= PTRDIFF_MAX, "Invalid allocation size: %.03fGiB, must be within (0, %.01fGiB]", (double)len/(double)(1<<30), (double)PTRDIFF_MAX/(double)(1<<30));
    if (ctx->delta - ctx->chunks[ctx->chunks_len - 1] < (ptrdiff_t)len) {
        if (ctx->chunk_size < len) { // Increase the chunk size if it's too small to accommodate the requested length
            do ctx->chunk_size <<= 1;
            while (ctx->chunk_size < len && ctx->chunk_size <= (PTRDIFF_MAX>>1));
        }
        pt_ctx_push_chunk(ctx);
        pt_log_error(
            "Pool chunk exhausted - requested %.03fKiB\n"
            "Increase pool chunk size for best performance, current pool chunk size: %.03fGiB, total allocated: %.03fGiB",
            (double)len/(double)(1<<10),
            (double)ctx->chunk_size/(double)(1<<30),
            (double)(ctx->chunk_size*ctx->chunks_len)/(double)(1<<30)
        );
    }
    ctx->delta -= len;
    uint8_t *const p = ctx->delta;
    ++ctx->alloc_acc;
    ctx->alloc_total += len;
    return p;
}

void pt_ctx_free(struct pt_ctx_t *const ctx) {
    for (size_t i = 0; i < ctx->chunks_len; ++i)
        (*ctx->alloc)(ctx->chunks[i], 0);
    (*ctx->alloc)(ctx->chunks, 0);
    memset(ctx, 0, sizeof(*ctx));
}

struct pt_tensor_t *pt_tensor_new(struct pt_ctx_t *ctx, const pt_dim_t *const dims, const pt_dim_t num_dims) {
    assert(num_dims > 0 && num_dims <= PT_MAX_DIMS);
    pt_dim_t bytes = sizeof(float);
    for (pt_dim_t i = 0; i < num_dims; ++i) // Accumulate the total data size in bytes
        bytes *= dims[i];
    struct pt_tensor_t *tensor = pt_ctx_pool_alloc(ctx, sizeof(*tensor) + bytes); // Allocate memory for the tensor + data
    memset(tensor, 0, sizeof(*tensor));
    tensor->ctx = ctx;
    tensor->size = bytes;
    tensor->data = (float *)(tensor + 1); // Set the data pointer to the end of the tensor structure, where the data follows
    memset(tensor->data, 0, bytes); // Initialize the data to zero
    for (pt_dim_t i = 0; i < PT_MAX_DIMS; ++i) // Set dimensions and strides to identity to saturate out zero multiplication because: x * 0 = 0
        tensor->dims[i] = 1;
    memcpy(tensor->dims, dims, num_dims * sizeof(*dims));
    tensor->strides[0] = sizeof(*tensor->data);
    for (pt_dim_t i = 1; i < PT_MAX_DIMS; ++i) // Calculate strides for each dimension
        tensor->strides[i] = tensor->strides[i - 1] * tensor->dims[i - 1];
    return tensor;
}
