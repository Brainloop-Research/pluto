// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>
// Header file for basic linear algebra subprograms (BLAS) for Brainloop Research's Pluto library.
// Can be directly included for multiple implementations for different ISAs

#include "pt_core.h"

#ifdef __ARM_NEON
#   include <arm_neon.h>
#endif
#if defined(_MSC_VER) || defined(__MINGW32__)
#   include <intrin.h>
#elif defined(__x86_64__) || defined(_M_AMD64)
#   include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// TODO: Accelerate with intrinsics and runtime detection

struct pt_f16_t { uint16_t bits; }; // IEEE 754 754-2008 binary 16 (half precision float)
pt_static_assert(sizeof(struct pt_f16_t) == 2);

#define pt_f16_c(x) ((struct pt_f16_t){.bits=(x)&0xffff})
#define PT_F16_E pt_f16_c(0x4170)
#define PT_F16_EPSILON pt_f16_c(0x1400)
#define PT_F16_FRAC_1_PI pt_f16_c(0x3518)
#define PT_F16_FRAC_1_SQRT_2 pt_f16_c(0x39a8)
#define PT_F16_FRAC_2_PI pt_f16_c(0x3918)
#define PT_F16_FRAC_2_SQRT_PI pt_f16_c(0x3c83)
#define PT_F16_FRAC_PI_2 pt_f16_c(0x3e48)
#define PT_F16_FRAC_PI_3 pt_f16_c(0x3c30)
#define PT_F16_FRAC_PI_4 pt_f16_c(0x3a48)
#define PT_F16_FRAC_PI_6 pt_f16_c(0x3830)
#define PT_F16_FRAC_PI_8 pt_f16_c(0x3648)
#define PT_F16_INFINITY pt_f16_c(0x7c00)
#define PT_F16_LN_10 pt_f16_c(0x409b)
#define PT_F16_LN_2 pt_f16_c(0x398c)
#define PT_F16_LOG10_2 pt_f16_c(0x34d1)
#define PT_F16_LOG10_E pt_f16_c(0x36f3)
#define PT_F16_LOG2_10 pt_f16_c(0x42a5)
#define PT_F16_LOG2_E pt_f16_c(0x3dc5)
#define PT_F16_MAX pt_f16_c(0x7bff)
#define PT_F16_MAX_SUBNORMAL pt_f16_c(0x03ff)
#define PT_F16_MIN pt_f16_c(0xfbff)
#define PT_F16_MIN_POSITIVE pt_f16_c(0x0400)
#define PT_F16_MIN_POSITIVE_SUBNORMAL pt_f16_c(0x0001)
#define PT_F16_NAN pt_f16_c(0x7e00)
#define PT_F16_NEG_INFINITY pt_f16_c(0xfc00)
#define PT_F16_NEG_ONE pt_f16_c(0xbc00)
#define PT_F16_NEG_ZERO pt_f16_c(0x8000)
#define PT_F16_ONE pt_f16_c(0x3c00)
#define PT_F16_PI pt_f16_c(0x4248)
#define PT_F16_SQRT_2 pt_f16_c(0x3da8)
#define PT_F16_ZERO pt_f16_c(0x0000)

struct pt_bf16_t { uint16_t bits; }; // Google brain float 16 (bfloat16)
pt_static_assert(sizeof(struct pt_bf16_t) == 2);
#define pt_bf16_c(x) ((struct pt_bf16_t){.bits=(x)&0xffff})

#define PT_BF16_EPSILON pt_bf16_c(0x3c00)
#define PT_BF16_INFINITY pt_bf16_c(0x7f80)
#define PT_BF16_MAX pt_bf16_c(0x7f7f)
#define PT_BF16_MIN pt_bf16_c(0xff7f)
#define PT_BF16_MIN_POSITIVE pt_bf16_c(0x0080)
#define PT_BF16_NAN pt_bf16_c(0x7FC0)
#define PT_BF16_NEG_INFINITY pt_bf16_c(0xff80)
#define PT_BF16_MIN_POSITIVE_SUBNORMAL pt_bf16_c(0x0001)
#define PT_BF16_MAX_SUBNORMAL pt_bf16_c(0x007f)
#define PT_BF16_ONE pt_bf16_c(0x3f80)
#define PT_BF16_ZERO pt_bf16_c(0x0000)
#define PT_BF16_NEG_ZERO pt_bf16_c(0x8000)
#define PT_BF16_NEG_ONE pt_bf16_c(0xbf80)
#define PT_BF16_E pt_bf16_c(0x402e)
#define PT_BF16_PI pt_bf16_c(0x4049)
#define PT_BF16_FRAC_1_PI pt_bf16_c(0x3ea3)
#define PT_BF16_FRAC_1_SQRT_2 pt_bf16_c(0x3f35)
#define PT_BF16_FRAC_2_PI pt_bf16_c(0x3f23)
#define PT_BF16_FRAC_2_SQRT_PI pt_bf16_c(0x3f90)
#define PT_BF16_FRAC_PI_2 pt_bf16_c(0x3fC9)
#define PT_BF16_FRAC_PI_3 pt_bf16_c(0x3f86)
#define PT_BF16_FRAC_PI_4 pt_bf16_c(0x3f49)
#define PT_BF16_FRAC_PI_6 pt_bf16_c(0x3f06)
#define PT_BF16_FRAC_PI_8 pt_bf16_c(0x3eC9)
#define PT_BF16_LN_10 pt_bf16_c(0x4013)
#define PT_BF16_LN_2 pt_bf16_c(0x3f31)
#define PT_BF16_LOG10_E pt_bf16_c(0x3ede)
#define PT_BF16_LOG10_2 pt_bf16_c(0x3e9a)
#define PT_BF16_LOG2_E pt_bf16_c(0x3fb9)
#define PT_BF16_LOG2_10 pt_bf16_c(0x4055)
#define PT_BF16_SQRT_2 pt_bf16_c(0x3fb5)

static inline float pt_blas_cvt_f16_to_f32_sca(const struct pt_f16_t x) {
#if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
    return (float)*(__fp16 *)&x;
#else // Slow software emulated path
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
#endif
}

static inline struct pt_f16_t pt_blas_cvt_f32_to_f16_sca(const float x) {
#if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
    const __fp16 f16 = (__fp16)x;
    return *(struct pt_f16_t *)&f16;
#else // Slow software emulated path
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
    return (struct pt_f16_t){.bits=(uint16_t)((sign>>16) | (shl1_w > 0xff000000 ? 0x7e00 : nonsign))}; // Pack full bit pattern
#endif
}

static inline float pt_blas_cvt_bf16_to_f32_sca(const struct pt_bf16_t x) {
    const uint32_t tmp = (uint32_t)x.bits<<16;
    return *(float *)&tmp;
}

static inline struct pt_bf16_t pt_blas_cvt_f32_to_bf16_sca(const float x) { // Same as x86-64 ASM instruction: vcvtneps2bf16 from AMD Zen4.
    struct pt_bf16_t bf16;
    if (((*(uint32_t *)&x) & 0x7fffffff) > 0x7f800000) { // NaN
        bf16.bits = 64 | ((*(uint32_t *)&x)>>16); // quiet NaNs only
        return bf16;
    }
    if (!((*(uint32_t *)&x) & 0x7f800000)) { // Subnormals
        bf16.bits = ((*(uint32_t *)&x) & 0x80000000)>>16; // Flush to zero
        return bf16;
    }
    bf16.bits = ((*(uint32_t *)&x) + (0x7fff + (((*(uint32_t *)&x)>>16) & 1)))>>16; // Rounding and composing final bf16 value
    return bf16;
}

static inline void pt_blas_cvt_f16_to_f32_vec(const size_t n, float *const o, const struct pt_f16_t *const x) {

    for (size_t i = 0; i < n; ++i) {
        o[i] = pt_blas_cvt_f16_to_f32_sca(x[i]);
    }
}

static inline void pt_blas_cvt_f32_to_f16_vec(const size_t n, struct pt_f16_t *const o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = pt_blas_cvt_f32_to_f16_sca(x[i]);
    }
}

static inline void pt_blas_cvt_bf16_to_f32_vec(const size_t n, float *const o, const struct pt_bf16_t *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = pt_blas_cvt_bf16_to_f32_sca(x[i]);
    }
}

static inline void pt_blas_cvt_f32_to_bf16_vec(const size_t n, struct pt_bf16_t *const o, const float *const x) {
    for (size_t i = 0; i < n; ++i) {
        o[i] = pt_blas_cvt_f32_to_bf16_sca(x[i]);
    }
}

#ifdef __cplusplus
}
#endif
