// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "pluto.h"

#include <assert.h>
#include <math.h>
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

#define inject_enum(_, mnemonic, __, ___) mnemonic
const char *const pt_opcode_mnemonic[PT_OPC_MAX] = { pt_opdef(inject_enum, PT_ENUM_SEP) };
#undef inject_enum
#define inject_enum(_, __, desc, ___) desc
const char *const pt_opcode_desc[PT_OPC_MAX] = { pt_opdef(inject_enum, PT_ENUM_SEP) };
#undef inject_enum
#define inject_enum(_, __, ___, nargs) (255&nargs)
const uint8_t pt_opcode_arg_count[PT_OPC_MAX] = { pt_opdef(inject_enum, PT_ENUM_SEP) };
#undef inject_enum

#if !defined(__ARM_ARCH) && (defined(__x86_64__) || defined(_WIN64))
static inline void pt_cpuid(uint32_t (*const o)[4], const uint32_t x) {
#ifdef _MSC_VER
    __cpuidex(*o, x, 0);
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

#if defined(__APPLE__) && defined(__aarch64__)
static bool pt_sysctl(const char* const id, char *const o, const size_t osz) {
    size_t len;
    if (pt_unlikely(sysctlbyname(id, NULL, &len, NULL, 0) != 0))
        return false;
    char *const scratch = alloca(len+1);
    if (pt_unlikely(sysctlbyname(id, scratch, &len, NULL, 0) != 0))
        return false;
    scratch[len] = '\0';
    snprintf(o, osz, "%s", scratch);
    return true;
}
#endif

static void pt_query_os_name(struct pt_ctx_t *const ctx) {
#if defined(__APPLE__) && defined(__aarch64__)
    pt_sysctl("kern.version", ctx->os_name, sizeof(ctx->os_name));
#elif !defined(__ARM_ARCH) && (defined(__x86_64__) || defined(_WIN64))
    FILE *rel = fopen("/etc/os-release", "rt");
    if (!rel) {
        rel = fopen("/usr/lib/os-release", "rt");
        if (!rel) {
            rel = fopen("/etc/lsb-release", "rt");
            if (pt_unlikely(!rel)) {
#ifdef __linux__
                strcpy(ctx->os_name, "Linux");
#elif defined(__FreeBSD__)
                strcpy(ctx->os_name, "FreeBSD");
#elif defined(__OpenBSD__)
                strcpy(ctx->os_name, "OpenBSD");
#elif defined(__NetBSD__)
                strcpy(ctx->os_name, "NetBSD");
#elif defined(__DragonFly__)
                strcpy(ctx->os_name, "DragonFly");
#else
                strcpy(ctx->os_name, "Unknown");
#endif
                return;
            } else {
                char line[512] = {0}, name[64] = {0};
                unsigned long major = 0, minor = 0, patch = 0, build = 0;
                while (fgets(line, sizeof(line), rel)) {
                    if (strncmp(line, "DISTRIB_ID", sizeof("DISTRIB_ID")-1) == 0) {
                        char *const value = strchr(line, '=')+1;
                        strncpy(name, value, sizeof(name)-1);
                        name[strcspn(name, "\n")] = '\0'; // Remove newline character
                    } else if (strncmp(line, "DISTRIB_RELEASE", sizeof("DISTRIB_RELEASE")-1) == 0) {
                        char *marker = strchr(line, '=')+1;
                        major = strtoul(marker, &marker, 10);
                        minor = strtoul(marker+1, &marker, 10);
                        patch = strtoul(marker+1, &marker, 10);
                        build = strtoul(marker+1, NULL, 10);
                    } else if (strncmp(line, "DISTRIB_DESCRIPTION", sizeof("DISTRIB_DESCRIPTION")-1) == 0) {
                        const char *const start_idx = strchr(line, '"') + 1;
                        const char *const end_idx = strrchr(line, '"');
                        if (start_idx && end_idx && end_idx > start_idx) {
                            size_t length = end_idx - start_idx;
                            strncpy(name, start_idx, pt_min(length, sizeof(name)-1));
                            name[length] = '\0'; // Null-terminate the string
                        }
                    }
                }
                snprintf(ctx->os_name, sizeof(ctx->os_name), "%s %lu.%lu.%lu.%lu", name, major, minor, patch, build);
                return;
            }
        }
    }
    char line[512] = {0}, name[64] = {0};
    unsigned long major = 0, minor = 0, patch = 0, build = 0;
    while (fgets(line, sizeof(line), rel)) {
        if (!*name && (strncmp(line, "NAME", sizeof("NAME")-1) == 0 || strncmp(line, "PRETTY_NAME", sizeof("PRETTY_NAME")-1) == 0)) {
            strcpy(name, strchr(line, '=') + 1);
            name[strcspn(name, "\n")] = '\0';
        } else if (strncmp(line, "VERSION_ID", sizeof("VERSION_ID")-1) == 0) {
            char *marker = strchr(line, '=') + 1;
            if (marker[0] == '"') ++marker;
            major = strtoul(marker, &marker, 10);
            if (marker[0] && marker[0] != '"')
                minor = strtoul(marker+1, &marker, 10);
            if (marker[0] && marker[0] != '"')
                patch = strtoul(marker+1, &marker, 10);
            if (marker[0] && marker[0] != '"')
                build = strtoul(marker+1, NULL, 10);
        }
    }
    size_t len; // trim quotes
    if ((len = strlen(name)) == 0) return;
    if (name[len-1] == '"') name[len-1] = '\0'; // trim trailing quote
    if (*name == '"') memmove(name, name+1, len); // trim leading quote
    snprintf(ctx->os_name, sizeof(ctx->os_name), "%s %lu.%lu.%lu.%lu", name, major, minor, patch, build); // format full name
#else
    strcpy(ctx->os_name, "Unknown");
#endif
}

static void pt_query_cpu_name(struct pt_ctx_t *const ctx) {
#if defined(__APPLE__) && defined(__aarch64__)
    pt_sysctl("machdep.cpu.brand_string", ctx->cpu_name, sizeof(ctx->cpu_name));
#elif !defined(__ARM_ARCH) && (defined(__x86_64__) || defined(_WIN64))
    uint32_t regs[4];
    pt_assert2(sizeof(ctx->cpu_name) >= sizeof(regs));
    char *const name = ctx->cpu_name;
    pt_cpuid(&regs, 0x80000002);
    for (size_t i = 0; i < 4; ++i)
        *(uint32_t *)(name+(i<<2)) = regs[i];
    pt_cpuid(&regs, 0x80000003);
    for (size_t i = 0; i < 4; ++i)
        *(uint32_t *)(name+(i<<2)+16) = regs[i];
    pt_cpuid(&regs, 0x80000004);
    for (size_t i = 0; i < 4; ++i)
        *(uint32_t *)(name+(i<<2)+32) = regs[i];
#else
    strcpy(ctx->cpu_name, "Unknown");
#endif
}

static void pt_ctx_push_chunk(struct pt_ctx_t *const ctx) {
    uint8_t *const chunk = (*ctx->alloc)(NULL, ctx->chunk_size);
    ctx->mapped_total += ctx->chunk_size;
    if (ctx->chunks_len == ctx->chunks_cap)
        ctx->chunks = (*ctx->alloc)(ctx->chunks, (ctx->chunks_cap <<= 1) * sizeof(*ctx->chunks));
    ctx->chunks[ctx->chunks_len++] = chunk;
    ctx->delta = chunk + ctx->chunk_size;
}

void pt_ctx_init(struct pt_ctx_t *const ctx, const pt_alloc_proc_t alloc, const size_t chunk_size) {
    if (chunk_size > 1 && chunk_size < (1<<20))
        pt_log_error("Chunk size very small: %zu, set it to >= 1MiB for best performance", chunk_size);
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc = alloc ? alloc : &pt_default_allocator;
    ctx->chunk_size = chunk_size ? chunk_size : PT_CTX_CHUNK_SIZE;
    ctx->chunks_cap = PT_CTX_CHUNKS_CAP;
    ctx->chunks = (*ctx->alloc)(NULL, ctx->chunks_cap * sizeof(*ctx->chunks));
    pt_ctx_push_chunk(ctx);
    pt_query_os_name(ctx);
    pt_query_cpu_name(ctx);
    printf("OS: %s\n", ctx->os_name);
    printf("CPU: %s\n", ctx->cpu_name);
    fflush(stdout);
}

void *pt_ctx_pool_alloc(struct pt_ctx_t *const ctx, const size_t len) {
    pt_assert(len && len <= PTRDIFF_MAX, "Invalid allocation size: %.03fGiB, must be within (0, %.01fGiB]", (double)len/(double)(1<<30), (double)PTRDIFF_MAX/(double)(1<<30));
    if (ctx->delta - ctx->chunks[ctx->chunks_len-1] < (ptrdiff_t)len) {
        if (ctx->chunk_size < len) { // Increase the chunk size if it's too small to accommodate the requested length
            do ctx->chunk_size <<= 1;
            while (ctx->chunk_size < len && ctx->chunk_size <= (PTRDIFF_MAX>>1));
        }
        pt_ctx_push_chunk(ctx);
#if PT_CTX_POOL_LOG_ENABLE
        pt_log_error(
            "Pool chunk exhausted - requested %.03fKiB\n"
            "Increase pool chunk size for best performance, current pool chunk size: %.03fGiB, total allocated: %.03fGiB",
            (double)len/(double)(1<<10),
            (double)ctx->chunk_size/(double)(1<<30),
            (double)(ctx->chunk_size*ctx->chunks_len)/(double)(1<<30)
        );
#endif
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
        tensor->shape[i] = 1;
    memcpy(tensor->shape, dims, num_dims * sizeof(*dims));
    tensor->strides[0] = sizeof(*tensor->data);
    for (pt_dim_t i = 1; i < PT_MAX_DIMS; ++i) // Calculate strides for each dimension
        tensor->strides[i] = tensor->strides[i - 1] * tensor->shape[i - 1];
    tensor->rank = num_dims;
    return tensor;
}

struct pt_tensor_t *pt_tensor_new_1d(struct pt_ctx_t *const ctx, const pt_dim_t d1) { return pt_tensor_new(ctx, (pt_dim_t[]){d1}, 1); }
struct pt_tensor_t *pt_tensor_new_2d(struct pt_ctx_t *const ctx, const pt_dim_t d1, const pt_dim_t d2) { return pt_tensor_new(ctx, (pt_dim_t[]){d1, d2}, 2);}
struct pt_tensor_t *pt_tensor_new_3d(struct pt_ctx_t *const ctx, const pt_dim_t d1, const pt_dim_t d2, const pt_dim_t d3) { return pt_tensor_new(ctx, (pt_dim_t[]){d1, d2, d3}, 3); }
struct pt_tensor_t *pt_tensor_new_4d(struct pt_ctx_t *const ctx, const pt_dim_t d1, const pt_dim_t d2, const pt_dim_t d3, const pt_dim_t d4) { return pt_tensor_new(ctx, (pt_dim_t[]){d1, d2, d3, d4}, 4); }

struct pt_tensor_t *pt_tensor_isomorphic(struct pt_ctx_t *const ctx, const struct pt_tensor_t *const tensor) {
    struct pt_tensor_t *const iso = pt_tensor_new(ctx, tensor->shape, tensor->rank);
    return iso;
}

struct pt_tensor_t *pt_tensor_clone(struct pt_ctx_t *const ctx, const struct pt_tensor_t *const tensor) {
    struct pt_tensor_t *const iso = pt_tensor_isomorphic(ctx, tensor);
    memcpy(iso->data, tensor->data, tensor->size); // Copy data
    return iso;
}

pt_dim_t pt_tensor_num_elems(const struct pt_tensor_t *const tensor) {
    return tensor->size / (pt_dim_t)sizeof(float);
}

void pt_tensor_fill(struct pt_tensor_t *const tensor, const float x) {
    if (x == 0.0f) {
        memset(tensor->data, 0, tensor->size);
    } else {
        for (pt_dim_t i = 0; i < pt_tensor_num_elems(tensor); ++i) {
            tensor->data[i] = x;
        }
    }
}

void pt_tensor_fill_fn(struct pt_tensor_t *const tensor, float (*const f)(pt_dim_t)) {
    for (pt_dim_t i = 0; i < pt_tensor_num_elems(tensor); ++i) {
        tensor->data[i] = (*f)(i);
    }
}

bool pt_tensor_is_scalar(const struct pt_tensor_t *const tensor) {
    for (int i = 0; i < PT_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool pt_tensor_is_vector(const struct pt_tensor_t *const tensor) {
    for (int i = 1; i < PT_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool pt_tensor_is_matrix(const struct pt_tensor_t *const tensor) {
    for (int i = 2; i < PT_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool pt_tensor_is_transposed(const struct pt_tensor_t *const tensor) {
    return tensor->shape[0] < tensor->shape[1];
}

bool pt_tensor_is_matmul_compatible(const struct pt_tensor_t *const a, const struct pt_tensor_t *const b) {
    return a->shape[1] == b->shape[0];
}

static void PT_UNUSED pt_blas_v_softmax(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = expf(x[i]);
    }
}
#define pt_blas_v_softmax_dv pt_blas_v_softmax // Derivative of e^x is e^x

static void PT_UNUSED pt_blas_v_sigmoid(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}
static void PT_UNUSED pt_blas_v_sigmoid_dv(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        const float y = 1.0f / (1.0f + expf(-x[i]));
        o[i] = y * (1.0f - y);
    }
}

static void PT_UNUSED pt_blas_v_relu(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = fmaxf(0.0f, x[i]);
    }
}
static void PT_UNUSED pt_blas_v_relu_dv(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = x[i] > 0.0f ? 1.0f : 0.0f;
    }
}
