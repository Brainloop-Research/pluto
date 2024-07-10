// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#ifndef PT_TENSOR_H
#define PT_TENSOR_H

#include "pt_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t pt_dim_t;
#define PT_DIM_MAX INT64_MAX
#define PT_DIM_MIN 0
#define PT_DIM_FMT PRIi64
#define PT_MAX_DIMS 4

struct pt_tensor_t {                // Structure to represent a tensor
    struct pt_ctx_t *ctx;           // Context host
    float *data;                    // Pointer to the data
    pt_dim_t shape[PT_MAX_DIMS];    // Size of each dimension
    pt_dim_t strides[PT_MAX_DIMS];  // Strides for each dimension
    pt_dim_t rank;                  // Number of dimensions
    pt_dim_t size;                  // Total size of data in bytes
    void *ud;                       // User data
};

extern PT_EXPORT struct pt_tensor_t *pt_tensor_new(struct pt_ctx_t *ctx, const pt_dim_t *dims, pt_dim_t num_dims);
extern PT_EXPORT struct pt_tensor_t *pt_tensor_new_1d(struct pt_ctx_t *ctx, pt_dim_t d1);
extern PT_EXPORT struct pt_tensor_t *pt_tensor_new_2d(struct pt_ctx_t *ctx, pt_dim_t d1, pt_dim_t d2);
extern PT_EXPORT struct pt_tensor_t *pt_tensor_new_3d(struct pt_ctx_t *ctx, pt_dim_t d1, pt_dim_t d2, pt_dim_t d3);
extern PT_EXPORT struct pt_tensor_t *pt_tensor_new_4d(struct pt_ctx_t *ctx, pt_dim_t d1, pt_dim_t d2, pt_dim_t d3, pt_dim_t d4);
extern PT_EXPORT struct pt_tensor_t *pt_tensor_isomorphic(struct pt_ctx_t *ctx, const struct pt_tensor_t *tensor);
extern PT_EXPORT struct pt_tensor_t *pt_tensor_clone(struct pt_ctx_t *ctx, const struct pt_tensor_t *tensor);
extern PT_EXPORT pt_dim_t pt_tensor_num_elems(const struct pt_tensor_t *tensor);
extern PT_EXPORT void pt_tensor_fill(struct pt_tensor_t *tensor, float x);
extern PT_EXPORT void pt_tensor_fill_fn(struct pt_tensor_t *tensor, float (*f)(pt_dim_t i));

extern PT_EXPORT bool pt_tensor_is_scalar(const struct pt_tensor_t *tensor);
extern PT_EXPORT bool pt_tensor_is_vector(const struct pt_tensor_t *tensor);
extern PT_EXPORT bool pt_tensor_is_matrix(const struct pt_tensor_t *tensor);
extern PT_EXPORT bool pt_tensor_is_transposed(const struct pt_tensor_t *tensor);
extern PT_EXPORT bool pt_tensor_is_matmul_compatible(const struct pt_tensor_t *a, const struct pt_tensor_t *b);

#ifdef __cplusplus
}
#endif

#endif // PT_TENSOR_H
