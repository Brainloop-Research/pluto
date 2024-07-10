// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

static const double f16_epsilon {pt_blas_cvt_f16_to_f32_sca(PT_F16_EPSILON)};
static const double bf16_epsilon {pt_blas_cvt_bf16_to_f32_sca(PT_BF16_EPSILON)};

static constexpr std::array<pt_f16_t, 16> f16_vec {
    PT_F16_E,
    PT_F16_PI,
    PT_F16_LOG2_E,
    PT_F16_ONE,
    PT_F16_E,
    PT_F16_PI,
    PT_F16_LOG2_E,
    PT_F16_ONE,
    PT_F16_E,
    PT_F16_PI,
    PT_F16_LOG2_E,
    PT_F16_ONE,
    PT_F16_E,
    PT_F16_PI,
    PT_F16_LN_2,
    PT_F16_LN_10,
};
static constexpr std::array<pt_bf16_t, 16> bf16_vec {
    PT_BF16_E,
    PT_BF16_PI,
    PT_BF16_LOG2_E,
    PT_BF16_ONE,
    PT_BF16_E,
    PT_BF16_PI,
    PT_BF16_LOG2_E,
    PT_BF16_ONE,
    PT_BF16_E,
    PT_BF16_PI,
    PT_BF16_LOG2_E,
    PT_BF16_ONE,
    PT_BF16_E,
    PT_BF16_PI,
    PT_BF16_LN_2,
    PT_BF16_LN_10,
};
static constexpr std::array<float, 16> f32_vec {
    M_E,
    M_PI,
    M_LOG2E,
    1.0,
    M_E,
    M_PI,
    M_LOG2E,
    1.0,
    M_E,
    M_PI,
    M_LOG2E,
    1.0,
    M_E,
    M_PI,
    M_LN2,
    M_LN10,
};

GTEST_TEST(blas, cvt_f16_to_f32) {
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_E) - M_E), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_PI) - M_PI), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_LOG2_E) - M_LOG2E), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_ONE) - 1.0), f16_epsilon);
    ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(PT_F16_ZERO) - 0.0), f16_epsilon);
    ASSERT_TRUE(std::isnan(pt_blas_cvt_f16_to_f32_sca(PT_F16_NAN)));
    ASSERT_FALSE(std::isnan(pt_blas_cvt_f16_to_f32_sca(PT_F16_ONE)));
}

GTEST_TEST(blas, cvt_f16_to_f32_vec) {
    std::array<float, 16> f32_out {};
    pt_blas_cvt_f16_to_f32_vec(f16_vec.size(), f32_out.data(), f16_vec.data());
    for (size_t i {0}; i < f32_vec.size(); ++i) {
        ASSERT_LE(std::abs(f32_out[i] - f32_vec[i]), f16_epsilon);
    }
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

GTEST_TEST(blas, cvt_f32_to_f16_vec) {
    std::array<pt_f16_t, 16> f16_out {};
    pt_blas_cvt_f32_to_f16_vec(f32_vec.size(), f16_out.data(), f32_vec.data());
    for (size_t i {0}; i < f16_vec.size(); ++i) {
        ASSERT_LE(std::abs(pt_blas_cvt_f16_to_f32_sca(f16_out[i]) - f32_vec[i]), f16_epsilon);
    }
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

GTEST_TEST(blas, cvt_bf16_to_f32_vec) {
    std::array<float, 16> f32_out {};
    pt_blas_cvt_bf16_to_f32_vec(bf16_vec.size(), f32_out.data(), bf16_vec.data());
    for (size_t i {0}; i < f32_vec.size(); ++i) {
        ASSERT_LE(std::abs(f32_out[i] - f32_vec[i]), bf16_epsilon);
    }
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

GTEST_TEST(blas, cvt_f32_to_bf16_vec) {
    std::array<pt_bf16_t, 16> bf16_out {};
    pt_blas_cvt_f32_to_bf16_vec(f32_vec.size(), bf16_out.data(), f32_vec.data());
    for (size_t i {0}; i < bf16_vec.size(); ++i) {
        ASSERT_LE(std::abs(pt_blas_cvt_bf16_to_f32_sca(bf16_out[i]) - f32_vec[i]), bf16_epsilon);
    }
}
