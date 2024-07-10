// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#include "pt_blas.h"

static void PT_UNUSED PT_AINLINE pt_blas_v_softmax_f32(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = expf(x[i]);
    }
}
#define pt_blas_v_softmax_dv_f32 pt_blas_v_softmax_f32 // D/Dx(e^x) = e^x

static void PT_UNUSED PT_AINLINE pt_blas_v_sigmoid_f32(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}
static void PT_UNUSED PT_AINLINE pt_blas_v_sigmoid_dv_f32(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        float y;
        pt_blas_v_sigmoid_f32(1, &y, x+i);
        o[i] = y * (1.0f - y);
    }
}

static void PT_UNUSED PT_AINLINE pt_blas_v_relu_f32(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = fmaxf(0.0f, x[i]);
    }
}
static void PT_UNUSED PT_AINLINE pt_blas_v_relu_dv_f32(const size_t n, float *o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = x[i] > 0.0f ? 1.0f : 0.0f;
    }
}
