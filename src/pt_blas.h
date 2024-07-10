// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#ifndef PT_BLAS_H
#define PT_BLAS_H

#include "pt_core.h"

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

extern PT_EXPORT float pt_blas_cvt_f16_to_f32_sca(struct pt_f16_t x);
extern PT_EXPORT struct pt_f16_t pt_blas_cvt_f32_to_f16_sca(float x);
extern PT_EXPORT float pt_blas_cvt_bf16_to_f32_sca(struct pt_bf16_t x);
extern PT_EXPORT struct pt_bf16_t pt_blas_cvt_f32_to_bf16_sca(float x);

extern PT_EXPORT void pt_blas_cvt_f16_to_f32_vec(size_t n, float *o, const struct pt_f16_t *x);
extern PT_EXPORT void pt_blas_cvt_f32_to_f16_vec(size_t n, struct pt_f16_t *o, const float *x);
extern PT_EXPORT void pt_blas_cvt_bf16_to_f32_vec(size_t n, float *o, const struct pt_bf16_t *x);
extern PT_EXPORT void pt_blas_cvt_f32_to_bf16_vec(size_t n, struct pt_bf16_t *o, const float *x);

#ifdef __cplusplus
}
#endif

#endif
