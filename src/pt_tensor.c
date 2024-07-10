// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#include "pt_tensor.h"

#include <stdarg.h>

struct pt_tensor_t *pt_tensor_new(struct pt_ctx_t *ctx, const pt_dim_t *const dims, const pt_dim_t num_dims) {
    pt_assert(num_dims > 0 && num_dims <= PT_MAX_DIMS, "Number of dimensions %" PT_DIM_FMT " must be within (1, %" PT_DIM_FMT ")", num_dims, PT_MAX_DIMS);
    pt_dim_t bytes = sizeof(float);
    for (pt_dim_t i = 0; i < num_dims; ++i) // Accumulate the total data size in bytes
        bytes *= dims[i];
    struct pt_tensor_t *tensor = pt_ctx_pool_alloc(ctx, sizeof(*tensor) + bytes); // Allocate memory for the tensor + data
    memset(tensor, 0, sizeof(*tensor));
    tensor->ctx = ctx;
    tensor->size = bytes;
    tensor->buf = (float *)(tensor + 1); // Set the data pointer to the end of the tensor structure, where the data follows
    memset(tensor->buf, 0, bytes); // Initialize the data to zero
    for (pt_dim_t i = 0; i < PT_MAX_DIMS; ++i) // Set dimensions and strides to identity to saturate out zero multiplication because: x * 0 = 0
        tensor->shape[i] = 1;
    memcpy(tensor->shape, dims, num_dims * sizeof(*dims));
    tensor->strides[0] = sizeof(*tensor->buf);
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
    memcpy(iso->buf, tensor->buf, tensor->size); // Copy data
    return iso;
}

pt_dim_t pt_tensor_element_count(const struct pt_tensor_t *const tensor) {
    return tensor->size / (pt_dim_t)sizeof(float);
}

pt_dim_t pt_tensor_row_count(const struct pt_tensor_t *const tensor) {
    pt_dim_t prod = 1;
    for (pt_dim_t i = 1; i < PT_MAX_DIMS; ++i)
        prod *= tensor->shape[i];
    return prod;
}

pt_dim_t pt_tensor_column_count(const struct pt_tensor_t *const tensor) {
    return tensor->shape[0];
}

void pt_linear_to_multidim_idx(const struct pt_tensor_t *const tensor, const pt_dim_t i, pt_dim_t (*const dims)[4]) {
    const pt_dim_t d0 = tensor->shape[0];
    const pt_dim_t d1 = tensor->shape[1];
    const pt_dim_t d2 = tensor->shape[2];
    (*dims)[3] = i / (d2*d1*d0);
    (*dims)[2] = (i - (*dims)[3]*d2*d1*d0) / (d1*d0);
    (*dims)[1] = (i - (*dims)[3]*d2*d1*d0 - (*dims)[2]*d1*d0) / d0;
    (*dims)[0] =  i - (*dims)[3]*d2*d1*d0 - (*dims)[2]*d1*d0 - (*dims)[1]*d0;
}

void pt_tensor_fill(struct pt_tensor_t *const tensor, const float x) {
    if (x == 0.0f) {
        memset(tensor->buf, 0, tensor->size);
    } else {
        for (pt_dim_t i = 0; i < pt_tensor_element_count(tensor); ++i) {
            tensor->buf[i] = x;
        }
    }
}

void pt_tensor_fill_fn(struct pt_tensor_t *const tensor, float (*const f)(pt_dim_t)) {
    for (pt_dim_t i = 0; i < pt_tensor_element_count(tensor); ++i) {
        tensor->buf[i] = (*f)(i);
    }
}

void pt_tensor_set_name(struct pt_tensor_t *const tensor, const char *const name) {
    strncpy(tensor->name, name, sizeof(tensor->name));
    tensor->name[(sizeof(tensor->name)/sizeof(*tensor->name))-1] = '\0';
}

void pt_tensor_fmt_name(struct pt_tensor_t *const tensor, const char *const fmt, ...) {
    va_list ar;
    va_start(ar, fmt);
    vsnprintf(tensor->name, sizeof(tensor->name) / sizeof(*tensor->name), fmt, ar);
    va_end(ar);
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

bool pt_tensor_is_higherorder3d(const struct pt_tensor_t *tensor) {
    return tensor->shape[PT_MAX_DIMS-1] == 1;
}

bool pt_tensor_is_transposed(const struct pt_tensor_t *const tensor) {
    return tensor->shape[0] < tensor->shape[1];
}

bool pt_tensor_is_matmul_compatible(const struct pt_tensor_t *const a, const struct pt_tensor_t *const b) {
    return a->shape[1] == b->shape[0];
}

bool pt_tensor_is_contiguous(const struct pt_tensor_t *const tensor) {
    if (tensor->strides[0] != sizeof(float))
        return false;
    for (int i = 1; i < PT_MAX_DIMS; ++i)
        if (tensor->strides[i] != tensor->strides[i-1] * tensor->shape[i-1])
            return false;
    return true;
}

bool pt_tensor_can_repeat(const struct pt_tensor_t *const tensor, const struct pt_tensor_t *const other) {
    for (int i = 0; i < PT_MAX_DIMS; ++i)
        if (other->shape[i] % tensor->shape[i] != 0)
            return false;
    return true;
}

bool pt_tensor_is_shape_eq(const struct pt_tensor_t *const tensor, const struct pt_tensor_t *const other) {
    return tensor == other || (tensor->rank == other->rank && memcmp(tensor->shape, other->shape, PT_MAX_DIMS * sizeof(*tensor->shape)) == 0);
}

bool pt_tensor_is_stride_eq(const struct pt_tensor_t *const tensor, const struct pt_tensor_t *const other) {
    return tensor == other || (tensor->rank == other->rank && memcmp(tensor->strides, other->strides, PT_MAX_DIMS * sizeof(*tensor->strides)) == 0);
}
