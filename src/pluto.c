// (c) 2024 Brainloop Research

#include "pluto.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

void *pt_default_allocator(void *blk, const size_t len) {
    if (!len) {
        free(blk);
        return NULL;
    } else if(!blk) {
        blk = malloc(len);
        assert(blk);
        return blk;
    } else {
        void *block = realloc(blk, len);
        assert(block);
        return block;
    }
}

static void pt_ctx_push_chunk(pt_ctx_t *const ctx) {
    uint8_t *const chunk = (*ctx->alloc)(NULL, ctx->chunk_size);
    if (ctx->chunks_len == ctx->chunks_cap)
        ctx->chunks = (*ctx->alloc)(ctx->chunks, (ctx->chunks_cap <<= 1) * sizeof(*ctx->chunks));
    ctx->chunks[ctx->chunks_len++] = chunk;
    ctx->delta = chunk + ctx->chunk_size;
}

void pt_ctx_init(pt_ctx_t *const ctx, const pt_alloc_proc_t alloc, const size_t chunk_size) {
    if (chunk_size && chunk_size < (1 << 20))
        pt_warn("Chunk size very small, set it to >= 1MiB for best performance");
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc = alloc ? alloc : &pt_default_allocator;
    ctx->chunk_size = chunk_size ? chunk_size : PT_CTX_CHUNK_SIZE;
    ctx->chunks = (*ctx->alloc)(NULL, PT_CTX_CHUNKS_CAP * sizeof(*ctx->chunks));
    ctx->chunks_cap = PT_CTX_CHUNKS_CAP;
    ctx->chunks_len = 0;
    pt_ctx_push_chunk(ctx);
}

void *pt_ctx_pool_alloc(pt_ctx_t *const ctx, const size_t len) {
    if (ctx->delta - ctx->chunks[ctx->chunks_len - 1] < len) {
        pt_ctx_push_chunk(ctx);
        pt_warn(
            "Pool chunk exhausted - requested %.03fKiB\n"
            "Increase pool chunk size for best performance, current pool chunk size: %.03fGiB, total allocated: %.03fGiB",
            (double)len/(double)(1<<10),
            (double)ctx->chunk_size/(double)(1<<30),
            (double)(ctx->chunk_size*ctx->chunks_len)/(double)(1<<30)
        );
    }
    uint8_t *const p = ctx->delta;
    ctx->delta -= len;
    return p;
}

void pt_ctx_free(pt_ctx_t *const ctx) {
    for (size_t i = 0; i < ctx->chunks_len; ++i)
        (*ctx->alloc)(ctx->chunks[i], 0);
    (*ctx->alloc)(ctx->chunks, 0);
    memset(ctx, 0, sizeof(*ctx));
}

pt_tensor_t *pt_tensor_new(const pt_dim_t *const dims, const pt_dim_t num_dims) {
    assert(num_dims > 0 && num_dims <= PT_MAX_DIMS);
    pt_dim_t bytes = sizeof(float);
    for (pt_dim_t i = 0; i < num_dims; ++i) // Accumulate the total data size in bytes
        bytes *= dims[i];
    pt_tensor_t *tensor = malloc(sizeof(*tensor) + bytes); // Allocate memory for the tensor + data
    memset(tensor, 0, sizeof(*tensor));
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

int main() {
    pt_ctx_t ctx;
    pt_ctx_init(&ctx, NULL, 0);
    for (int i = 0; i < 10; ++i) {
        int *a = pt_ctx_pool_alloc(&ctx, i * sizeof(*a));
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
    }
    pt_ctx_free(&ctx);
    return 0;
}
