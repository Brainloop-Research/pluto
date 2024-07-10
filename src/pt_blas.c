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

float pt_blas_cvt_f16_to_f32_sca(const struct pt_f16_t x) {
    const uint32_t w = (uint32_t)x.bits<<16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = 0xe0u<<23; // Exponent offset for normalization
    uint32_t tmp = (two_w>>4) + exp_offset; // Adjust exponent
    const float norm_x = *(float *)&tmp * 0x1.0p-112f; // Normalize the result
    tmp = (two_w>>17) | (126u<<23); // Adjust exponent for denormalized values
    const float denorm_x = *(float *)&tmp - 0.5f;
    const uint32_t denorm_cutoff = 1u<<27; // Threshold for denormalized values
    const uint32_t result = sign // Combine sign and mantissa
        | (two_w < denorm_cutoff
        ? *(uint32_t *)&denorm_x // Use denormalized value if below cutoff
        : *(uint32_t *)&norm_x); // Else use normalized value
    return *(float *)&result;
}

struct pt_f16_t pt_blas_cvt_f32_to_f16_sca(const float x) {
    float base = (fabsf(x) * 0x1.0p+112f) * 0x1.0p-110f;  // Normalize |x|
    const uint32_t w = *(uint32_t *)&x;
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xff000000u; // Extract bias
    if (bias < 0x71000000u) bias = 0x71000000u; // Apply minimum bias for subnormals
    bias = (bias>>1) + 0x07800000u; // Adjust bias for half precision
    base = *(float *)&bias + base;
    const uint32_t bits = *(uint32_t *)&base; // Extract bits
    const uint32_t exp_bits = (bits>>13) & 0x00007c00u; // Extract exponent bits
    const uint32_t mant_bits = bits & 0x00000fffu; // Extract mantissa bits
    const uint32_t nonsign = exp_bits + mant_bits; // Combine exponent and mantissa bits
    return (struct pt_f16_t){.bits=(sign>>16) | (shl1_w > 0xff000000 ? 0x7e00 : nonsign)}; // Pack full bit pattern
}

float pt_blas_cvt_bf16_to_f32_sca(const struct pt_bf16_t x) {
    const uint32_t tmp = (uint32_t)x.bits<<16;
    return *(float *)&tmp;
}

struct pt_bf16_t pt_blas_cvt_f32_to_bf16_sca(const float x) { // Same as x86-64 ASM instruction: vcvtneps2bf16 from AMD Zen4.
    struct pt_bf16_t bf16;
    if (((*(uint32_t *)&x) & 0x7fffffff) > 0x7f800000) { // NaN
        bf16.bits = 64 | ((*(uint32_t *)&x)>>16); // quiet NaNs only
        return bf16;
    }
    if (!((*(uint32_t *)&x) & 0x7f800000)) { // Subnormals
        bf16.bits = ((*(uint32_t *)&x) & 0x80000000)>>16; // Flush to zero
        return bf16;
    }
    bf16.bits = ((*(uint32_t*)&x) + (0x7fff + (((*(uint32_t *)&x)>>16) & 1)))>>16; // Rounding and composing final bf16 value
    return bf16;
}

void pt_blas_cvt_f16_to_f32_vec(const size_t n, float *const o, const struct pt_f16_t *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = pt_blas_cvt_f16_to_f32_sca(x[i]);
    }
}

void pt_blas_cvt_f32_to_f16_vec(const size_t n, struct pt_f16_t *const o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = pt_blas_cvt_f32_to_f16_sca(x[i]);
    }
}

void pt_blas_cvt_bf16_to_f32_vec(const size_t n, float *const o, const struct pt_bf16_t *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = pt_blas_cvt_bf16_to_f32_sca(x[i]);
    }
}

void pt_blas_cvt_f32_to_bf16_vec(const size_t n, struct pt_bf16_t *const o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = pt_blas_cvt_f32_to_bf16_sca(x[i]);
    }
}
