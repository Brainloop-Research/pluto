// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include <numeric>
#include "prelude.hpp"

using namespace pluto;

static constexpr std::array<f16, 16> f16_vec {
    f16::e(),
    f16::pi(),
    f16::log2_e(),
    f16::one(),
    f16::e(),
    f16::pi(),
    f16::log2_e(),
    f16::one(),
    f16::e(),
    f16::pi(),
    f16::log2_e(),
    f16::one(),
    f16::e(),
    f16::pi(),
    f16::log2_e(),
    f16::one(),
};
static constexpr std::array<bf16, 16> bf16_vec {
    bf16::e(),
    bf16::pi(),
    bf16::log2_e(),
    bf16::one(),
    bf16::e(),
    bf16::pi(),
    bf16::log2_e(),
    bf16::one(),
    bf16::e(),
    bf16::pi(),
    bf16::log2_e(),
    bf16::one(),
    bf16::e(),
    bf16::pi(),
    bf16::log2_e(),
    bf16::one(),
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
    M_LOG2E,
    1.0
};

static_assert(f16_vec.size() == bf16_vec.size());
static_assert(f16_vec.size() == f32_vec.size());

GTEST_TEST(blas, cvt_f16_to_f32) {
    for (std::size_t i {}; i < f16_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(static_cast<float>(f16_vec[i]) - f32_vec[i])
            < static_cast<float>(f16::eps())); // |ξ1 - ξ2| < ε
    }
    ASSERT_TRUE(std::isnan(static_cast<float>(f16::nan())));
    ASSERT_FALSE(std::isnan(static_cast<float>(f16::e())));
}

GTEST_TEST(blas, cvt_f32_to_f16) {
    for (std::size_t i {}; i < f16_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(static_cast<float>(f16{f32_vec[i]})
            - static_cast<float>(f16_vec[i]))
            < static_cast<float>(f16::eps())); // |ξ1 - ξ2| < ε
    }
    ASSERT_TRUE(std::isnan(static_cast<float>(f16{std::numeric_limits<float>::quiet_NaN()})));
    ASSERT_FALSE(std::isnan(static_cast<float>(f16{static_cast<float>(M_E)})));
}

GTEST_TEST(blas, cvt_f16_to_f32_vec) {
    std::array<float, 16> f32_out {};
    f16::cvt_f16_to_f32_vec(f16_vec.size(), f32_out.data(), f16_vec.data());
    for (std::size_t i {0}; i < f32_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(f32_out[i] - f32_vec[i])
            < static_cast<float>(f16::eps())); // |ξ1 - ξ2| < ε
    }
}

GTEST_TEST(blas, cvt_f32_to_f16_vec) {
    std::array<f16, 16> f16_out {};
    f16::cvt_f32_to_f16_vec(f32_vec.size(), f16_out.data(), f32_vec.data());
    for (std::size_t i {0}; i < f32_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(static_cast<float>(f16{f32_vec[i]})
             - static_cast<float>(f16_out[i]))
            < static_cast<float>(f16::eps())); // |ξ1 - ξ2| < ε
    }
}

GTEST_TEST(blas, cvt_bf16_to_f32) {
    for (std::size_t i {}; i < bf16_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(static_cast<float>(bf16_vec[i]) - f32_vec[i])
                    < static_cast<float>(bf16::eps())); // |ξ1 - ξ2| < ε
    }
    ASSERT_TRUE(std::isnan(static_cast<float>(bf16::nan())));
    ASSERT_FALSE(std::isnan(static_cast<float>(bf16::e())));
}

GTEST_TEST(blas, cvt_f32_to_bf16) {
    for (std::size_t i {}; i < bf16_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(static_cast<float>(bf16{f32_vec[i]})
                             - static_cast<float>(bf16_vec[i]))
                    < static_cast<float>(bf16::eps())); // |ξ1 - ξ2| < ε
    }
    ASSERT_TRUE(std::isnan(static_cast<float>(bf16{std::numeric_limits<float>::quiet_NaN()})));
    ASSERT_FALSE(std::isnan(static_cast<float>(bf16{static_cast<float>(M_E)})));
}

GTEST_TEST(blas, cvt_bf16_to_f32_vec) {
    std::array<float, 16> f32_out {};
    bf16::cvt_bf16_to_f32_vec(bf16_vec.size(), f32_out.data(), bf16_vec.data());
    for (std::size_t i {0}; i < f32_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(f32_out[i] - f32_vec[i])
                    < static_cast<float>(bf16::eps())); // |ξ1 - ξ2| < ε
    }
}

GTEST_TEST(blas, cvt_f32_to_bf16_vec) {
    std::array<bf16, 16> bf16_out {};
    bf16::cvt_f32_to_bf16_vec(f32_vec.size(), bf16_out.data(), f32_vec.data());
    for (std::size_t i {0}; i < f32_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(static_cast<float>(bf16{f32_vec[i]})
                             - static_cast<float>(bf16_out[i]))
                    < static_cast<float>(bf16::eps())); // |ξ1 - ξ2| < ε
    }
}

GTEST_TEST(vblas, dot_f32) {
    std::vector<float> data {};
    data.reserve(325);
    for (int i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i));
    }
    const float dot = vblas::dot(data.size(), data.data(), data.data());
    const float dot_ref = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f);
    ASSERT_FLOAT_EQ(dot, dot_ref);
}
