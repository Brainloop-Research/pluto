// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

static const double f16_epsilon {pt_blas_cvt_f16_to_f32_sca(PT_F16_EPSILON)};
static const double bf16_epsilon {pt_blas_cvt_bf16_to_f32_sca(PT_BF16_EPSILON)};

GTEST_TEST(blas, cvt_f16_to_f32) {
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_E) - M_E), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_PI) - M_PI), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_LOG2_E) - M_LOG2E), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_ONE) - 1.0), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_ZERO) - 0.0), f16_epsilon);
    ASSERT_TRUE(std::isnan(pt_blas_cvt_f16_to_f32_sca(PT_F16_NAN)));
    ASSERT_FALSE(std::isnan(pt_blas_cvt_f16_to_f32_sca(PT_F16_ONE)));
}

GTEST_TEST(blas, cvt_f32_to_f16) {
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(pt_blas_cvt_f32_to_f16_sca(M_E)) - M_E), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(pt_blas_cvt_f32_to_f16_sca(M_PI)) - M_PI), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(pt_blas_cvt_f32_to_f16_sca(M_LOG2E)) - M_LOG2E), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(pt_blas_cvt_f32_to_f16_sca(1.0)) - 1.0), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(pt_blas_cvt_f32_to_f16_sca(.0)) - 0.0), f16_epsilon);
    ASSERT_TRUE(std::isnan(pt_blas_cvt_f16_to_f32_sca(pt_blas_cvt_f32_to_f16_sca(std::nan("")))));
    ASSERT_FALSE(std::isnan(pt_blas_cvt_f16_to_f32_sca(pt_blas_cvt_f32_to_f16_sca(1.0))));
}

GTEST_TEST(blas, cvt_bf16_to_f32) {
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(PT_BF16_E) - M_E), bf16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(PT_BF16_PI) - M_PI), bf16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(PT_BF16_LOG2_E) - M_LOG2E), bf16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(PT_BF16_ONE) - 1.0), bf16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(PT_BF16_ZERO) - 0.0), bf16_epsilon);
    ASSERT_TRUE(std::isnan(pt_blas_cvt_bf16_to_f32_sca(PT_BF16_NAN)));
    ASSERT_FALSE(std::isnan(pt_blas_cvt_bf16_to_f32_sca(PT_BF16_ONE)));
}

GTEST_TEST(blas, cvt_f32_to_bf16) {
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(pt_blas_cvt_f32_to_bf16_sca(M_E)) - M_E), bf16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(pt_blas_cvt_f32_to_bf16_sca(M_PI)) - M_PI), bf16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(pt_blas_cvt_f32_to_bf16_sca(M_LOG2E)) - M_LOG2E), bf16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(pt_blas_cvt_f32_to_bf16_sca(1.0)) - 1.0), bf16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(pt_blas_cvt_f32_to_bf16_sca(.0)) - 0.0), bf16_epsilon);
    ASSERT_TRUE(std::isnan(pt_blas_cvt_bf16_to_f32_sca(pt_blas_cvt_f32_to_bf16_sca(std::nan("")))));
    ASSERT_FALSE(std::isnan(pt_blas_cvt_bf16_to_f32_sca(pt_blas_cvt_f32_to_bf16_sca(1.0))));
}
