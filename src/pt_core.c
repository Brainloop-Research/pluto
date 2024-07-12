// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#include "pt_core.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <time.h>

#ifdef __APPLE__
#   include <sys/sysctl.h>
#   include <sys/types.h>
#elif defined(__x86_64__) && !defined(_WIN32)
#   include <cpuid.h>
#   include <unistd.h>
#elif defined(_WIN32)
#   define WIN32_LEAN_AND_MEAN
#   include <windows.h>
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

#if defined(__x86_64__) || defined(_M_AMD64)
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
    ctx->boot_stamp = pt_hpc_micro_clock();
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
        printf(
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

uint64_t pt_hpc_micro_clock(void) {
#ifdef _WIN32
    static __int64 pt_timer_freq, pt_timer_start;
    LARGE_INTEGER li;
    if (!pt_timer_freq) {
        QueryPerformanceFrequency(&li);
        pt_timer_freq = li.QuadPart;
        QueryPerformanceCounter(&li);
        pt_timer_start = li.QuadPart;
    }
    QueryPerformanceCounter(&li);
    return (uint64_t)((li.QuadPart - pt_timer_start)*1000000ull / pt_timer_freq);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000000ull + (uint64_t)ts.tv_nsec/1000ull;
#endif
    return 0;
}
