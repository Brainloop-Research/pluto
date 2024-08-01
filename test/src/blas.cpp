// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include <numeric>

#include "prelude.hpp"

using namespace pluto;
using namespace blas;

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

GTEST_TEST(vblas, softmax_f32) {
    std::vector<float> data {};
    data.reserve(325);
    for (std::size_t i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i));
    }
    std::vector<float> r {};
    r.resize(data.size());
    detail::vblas::softmax(data.size(), r.data(), data.data());
    for (std::size_t i {}; i < data.size(); ++i) {
        ASSERT_EQ(r[i], std::exp(data[i]));
    }
}

GTEST_TEST(vblas, sigmoid_f32) {
    std::vector<float> data {};
    data.reserve(325);
    for (std::size_t i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i));
    }
    std::vector<float> r {};
    r.resize(data.size());
    detail::vblas::sigmoid(data.size(), r.data(), data.data());
    for (std::size_t i {}; i < data.size(); ++i) {
        ASSERT_EQ(r[i], 1.0f / (1.0f + std::exp(-data[i])));
    }
}

GTEST_TEST(vblas, tanh_f32) {
    std::vector<float> data {};
    data.reserve(325);
    for (std::size_t i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i));
    }
    std::vector<float> r {};
    r.resize(data.size());
    detail::vblas::tanh(data.size(), r.data(), data.data());
    for (std::size_t i {}; i < data.size(); ++i) {
        ASSERT_EQ(r[i], std::tanh(data[i]));
    }
}

GTEST_TEST(vblas, relu_f32) {
    using detail::vblas::gelu_coeff;
    using detail::vblas::sqrt2pi;
    std::vector<float> data {};
    data.reserve(325);
    for (std::size_t i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i % 2 == 0 ? -i : i));
    }
    std::vector<float> r {};
    r.resize(data.size());
    detail::vblas::relu(data.size(), r.data(), data.data());
    for (std::size_t i {}; i < data.size(); ++i) {
        ASSERT_EQ(r[i], std::max(0.0f, data[i]));
    }
}

GTEST_TEST(vblas, gelu_f32) {
    using detail::vblas::gelu_coeff;
    using detail::vblas::sqrt2pi;
    std::vector<float> data {};
    data.reserve(325);
    for (std::size_t i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i % 2 == 0 ? -i : i));
    }
    std::vector<float> r {};
    r.resize(data.size());
    detail::vblas::gelu(data.size(), r.data(), data.data());
    for (std::size_t i {}; i < data.size(); ++i) {
        ASSERT_EQ(r[i], 0.5f * data[i] * (1.0f + std::tanh(sqrt2pi * data[i] * (1.0f + gelu_coeff * data[i] * data[i]))));
    }
}

GTEST_TEST(vblas, add_f32) {
    std::vector<float> x {}, y {}, r {};
    std::generate_n(std::back_inserter(x), 325, []() noexcept -> float { return 1.0f; });
    std::generate_n(std::back_inserter(y), 325, []() noexcept -> float { return 2.0f; });
    r.resize(x.size());
    detail::vblas::add(x.size(), r.data(), x.data(), y.data());
    for (std::size_t i {}; i < x.size(); ++i) {
        ASSERT_FLOAT_EQ(r[i], x[i] + y[i]);
    }
}

GTEST_TEST(vblas, sub_f32) {
    std::vector<float> x {}, y {}, r {};
    std::generate_n(std::back_inserter(x), 325, []() { return 1.0f; });
    std::generate_n(std::back_inserter(y), 325, []() { return 2.0f; });
    r.resize(x.size());
    detail::vblas::sub(x.size(), r.data(), x.data(), y.data());
    for (std::size_t i {}; i < x.size(); ++i) {
        ASSERT_FLOAT_EQ(r[i], x[i] - y[i]);
    }
}

GTEST_TEST(vblas, mul_f32) {
    std::vector<float> x {}, y {}, r {};
    std::generate_n(std::back_inserter(x), 325, []() { return 1.0f; });
    std::generate_n(std::back_inserter(y), 325, []() { return 2.0f; });
    r.resize(x.size());
    detail::vblas::mul(x.size(), r.data(), x.data(), y.data());
    for (std::size_t i {}; i < x.size(); ++i) {
        ASSERT_FLOAT_EQ(r[i], x[i] * y[i]);
    }
}

GTEST_TEST(vblas, div_f32) {
    std::vector<float> x {}, y {}, r {};
    std::generate_n(std::back_inserter(x), 325, []() { return 1.0f; });
    std::generate_n(std::back_inserter(y), 325, []() { return 2.0f; });
    r.resize(x.size());
    detail::vblas::div(x.size(), r.data(), x.data(), y.data());
    for (std::size_t i {}; i < x.size(); ++i) {
        ASSERT_FLOAT_EQ(r[i], x[i] / y[i]);
    }
}

GTEST_TEST(vblas, dot_f32) {
    std::vector<float> data {};
    data.reserve(325);
    for (std::size_t i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i));
    }
    const float dot = detail::vblas::dot(data.size(), data.data(), data.data());
    const float dot_ref = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f);
    ASSERT_FLOAT_EQ(dot, dot_ref);
}

GTEST_TEST(blas, tensor_softmax) {
    constexpr float x1 {0.7f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    detail::vblas::softmax(1, &r1, &x1);
    t1->fill(x1);
    auto* r {softmax(compute_ctx{}, *t1)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_sigmoid) {
    constexpr float x1 {0.7f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    detail::vblas::sigmoid(1, &r1, &x1);
    t1->fill(x1);
    auto* r {sigmoid(compute_ctx{}, *t1)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_tanh) {
    constexpr float x1 {0.7f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    detail::vblas::tanh(1, &r1, &x1);
    t1->fill(x1);
    auto* r {tanh(compute_ctx{}, *t1)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_relu) {
    constexpr float x1 {0.7f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    detail::vblas::relu(1, &r1, &x1);
    t1->fill(x1);
    auto* r {relu(compute_ctx{}, *t1)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_gelu) {
    constexpr float x1 {0.7f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    detail::vblas::gelu(1, &r1, &x1);
    t1->fill(x1);
    auto* r {gelu(compute_ctx{}, *t1)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_silu) {
    constexpr float x1 {0.7f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    detail::vblas::silu(1, &r1, &x1);
    t1->fill(x1);
    auto* r {silu(compute_ctx{}, *t1)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_add_f32) {
    constexpr float x1 {1.0f}, x2 {2.0f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    auto* t2 {tensor::create(&ctx, {4*4, 4*8, 8*2, 2})};
    t1->fill(x1);
    t2->fill(x2);
    auto* r {add(compute_ctx{}, *t1, *t2)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, x1 + x2);
    }
}

GTEST_TEST(blas, tensor_sub_f32) {
    constexpr float x1 {1.0f}, x2 {2.0f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    auto* t2 {tensor::create(&ctx, {4*4, 4*8, 8*2, 2})};
    t1->fill(x1);
    t2->fill(x2);
    auto* r {sub(compute_ctx{}, *t1, *t2)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, x1 - x2);
    }
}

GTEST_TEST(blas, tensor_mul_f32) {
    constexpr float x1 {1.0f}, x2 {2.0f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    auto* t2 {tensor::create(&ctx, {4*4, 4*8, 8*2, 2})};
    t1->fill(x1);
    t2->fill(x2);
    auto* r {mul(compute_ctx{}, *t1, *t2)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, x1 * x2);
    }
}

GTEST_TEST(blas, tensor_div_f32) {
    constexpr float x1 {1.0f}, x2 {2.0f};
    context ctx {};
    auto* t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    auto* t2 {tensor::create(&ctx, {4*4, 4*8, 8*2, 2})};
    t1->fill(x1);
    t2->fill(x2);
    auto* r {div(compute_ctx{}, *t1, *t2)};
    ASSERT_TRUE(r->is_shape_eq(t1));
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, x1 / x2);
    }
}

static constexpr std::size_t M = 4, N = 16, K = 36;

// matrix A (MxK)
static constexpr std::array<float, M*K> matrix_a {
    2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
    10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
    7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
};

// matrix B (NxK)
static constexpr std::array<float, N*K> matrix_b {
    9.0f, 7.0f, 1.0f, 3.0f, 5.0f, 9.0f, 7.0f, 6.0f, 1.0f, 10.0f, 1.0f, 1.0f, 7.0f, 2.0f, 4.0f, 9.0f, 10.0f, 4.0f, 5.0f, 5.0f, 7.0f, 1.0f, 7.0f, 7.0f, 2.0f, 9.0f, 5.0f, 10.0f, 7.0f, 4.0f, 8.0f, 9.0f, 9.0f, 3.0f, 10.0f, 2.0f,
    4.0f, 6.0f, 10.0f, 9.0f, 5.0f, 1.0f, 8.0f, 7.0f, 4.0f, 7.0f, 2.0f, 6.0f, 5.0f, 3.0f, 1.0f, 10.0f, 8.0f, 4.0f, 8.0f, 3.0f, 7.0f, 1.0f, 2.0f, 7.0f, 6.0f, 8.0f, 6.0f, 5.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f, 5.0f, 7.0f, 1.0f,
    8.0f, 2.0f, 8.0f, 8.0f, 8.0f, 8.0f, 4.0f, 4.0f, 6.0f, 10.0f, 10.0f, 9.0f, 2.0f, 9.0f, 3.0f, 7.0f, 7.0f, 1.0f, 4.0f, 9.0f, 1.0f, 2.0f, 3.0f, 6.0f, 1.0f, 10.0f, 5.0f, 8.0f, 9.0f, 4.0f, 6.0f, 2.0f, 3.0f, 1.0f, 2.0f, 7.0f,
    5.0f, 1.0f, 7.0f, 2.0f, 9.0f, 10.0f, 9.0f, 5.0f, 2.0f, 5.0f, 4.0f, 10.0f, 9.0f, 9.0f, 1.0f, 9.0f, 8.0f, 8.0f, 9.0f, 4.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 8.0f, 4.0f, 5.0f, 10.0f, 7.0f, 6.0f, 2.0f, 1.0f, 10.0f, 10.0f, 7.0f,
    9.0f, 4.0f, 5.0f, 9.0f, 5.0f, 10.0f, 10.0f, 3.0f, 6.0f, 6.0f, 4.0f, 4.0f, 4.0f, 8.0f, 5.0f, 4.0f, 9.0f, 1.0f, 9.0f, 9.0f, 1.0f, 7.0f, 9.0f, 2.0f, 10.0f, 9.0f, 10.0f, 8.0f, 3.0f, 3.0f, 9.0f, 3.0f, 9.0f, 10.0f, 1.0f, 8.0f,
    9.0f, 2.0f, 6.0f, 9.0f, 7.0f, 2.0f, 3.0f, 5.0f, 3.0f, 6.0f, 9.0f, 7.0f, 3.0f, 7.0f, 6.0f, 4.0f, 10.0f, 3.0f, 5.0f, 7.0f, 2.0f, 9.0f, 3.0f, 2.0f, 2.0f, 10.0f, 8.0f, 7.0f, 3.0f, 10.0f, 6.0f, 3.0f, 1.0f, 1.0f, 4.0f, 10.0f,
    2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
    10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
    7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
    6.0f, 2.0f, 3.0f, 3.0f, 3.0f, 7.0f, 5.0f, 1.0f, 8.0f, 1.0f, 4.0f, 5.0f, 1.0f, 1.0f, 6.0f, 4.0f, 2.0f, 1.0f, 7.0f, 8.0f, 6.0f, 1.0f, 1.0f, 5.0f, 6.0f, 5.0f, 10.0f, 6.0f, 7.0f, 5.0f, 9.0f, 3.0f, 2.0f, 7.0f, 9.0f, 4.0f,
    2.0f, 5.0f, 9.0f, 5.0f, 10.0f, 3.0f, 1.0f, 8.0f, 1.0f, 7.0f, 1.0f, 8.0f, 1.0f, 6.0f, 7.0f, 8.0f, 4.0f, 9.0f, 5.0f, 10.0f, 3.0f, 7.0f, 6.0f, 8.0f, 8.0f, 5.0f, 6.0f, 8.0f, 10.0f, 9.0f, 4.0f, 1.0f, 3.0f, 3.0f, 4.0f, 7.0f,
    8.0f, 2.0f, 6.0f, 6.0f, 5.0f, 1.0f, 3.0f, 7.0f, 1.0f, 7.0f, 2.0f, 2.0f, 2.0f, 8.0f, 4.0f, 1.0f, 1.0f, 5.0f, 9.0f, 4.0f, 1.0f, 2.0f, 3.0f, 10.0f, 1.0f, 4.0f, 9.0f, 9.0f, 6.0f, 8.0f, 8.0f, 1.0f, 9.0f, 10.0f, 4.0f, 1.0f,
    8.0f, 5.0f, 8.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 1.0f, 9.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 1.0f, 4.0f, 6.0f, 7.0f, 7.0f, 7.0f, 8.0f, 7.0f, 8.0f, 8.0f, 2.0f, 10.0f, 2.0f, 7.0f, 3.0f, 8.0f, 3.0f,
    8.0f, 7.0f, 6.0f, 2.0f, 4.0f, 10.0f, 10.0f, 6.0f, 10.0f, 3.0f, 7.0f, 6.0f, 4.0f, 3.0f, 5.0f, 5.0f, 5.0f, 3.0f, 8.0f, 10.0f, 3.0f, 4.0f, 8.0f, 4.0f, 2.0f, 6.0f, 8.0f, 9.0f, 6.0f, 9.0f, 4.0f, 3.0f, 5.0f, 2.0f, 2.0f, 6.0f,
    10.0f, 6.0f, 2.0f, 1.0f, 7.0f, 5.0f, 6.0f, 4.0f, 1.0f, 9.0f, 10.0f, 2.0f, 4.0f, 5.0f, 8.0f, 5.0f, 7.0f, 4.0f, 7.0f, 6.0f, 3.0f, 9.0f, 2.0f, 1.0f, 4.0f, 2.0f, 6.0f, 6.0f, 3.0f, 3.0f, 2.0f, 8.0f, 5.0f, 9.0f, 3.0f, 4.0f,
};

// matrix C (MxN)
static constexpr std::array<float, M*N> matrix_c {
    1224.0f, 1023.0f, 1158.0f,1259.0f,1359.0f,1194.0f,1535.0f,1247.0f,1185.0f,1029.0f,889.0f,1182.0f,955.0f,1179.0f,1147.0f,1048.0f,
    1216.0f, 1087.0f, 1239.0f,1361.0f,1392.0f,1260.0f,1247.0f,1563.0f,1167.0f,1052.0f,942.0f,1214.0f,1045.0f,1134.0f,1264.0f,1126.0f,
    1125.0f, 966.0f, 1079.0f,1333.0f,1287.0f,1101.0f,1185.0f,1167.0f,1368.0f,990.0f,967.0f,1121.0f,971.0f,1086.0f,1130.0f,980.0f,
    999.0f, 902.0f, 1020.0f,1056.0f,1076.0f,929.0f,1029.0f,1052.0f,990.0f,1108.0f,823.0f,989.0f,759.0f,1041.0f,1003.0f,870.0f
};

GTEST_TEST(blas, tensor_sgemm_f32) {
    context ctx {};
    auto* t1 {tensor::create(&ctx, {M, K})};
    auto* t2 {tensor::create(&ctx, {N, K})};
    t1->populate(matrix_a);
    t2->populate(matrix_b);
    auto* r {matmul(compute_ctx{}, *t1, *t2)};
    ASSERT_EQ(r->buf().size(), matrix_c.size());
    for (std::size_t i {}; i < matrix_c.size(); ++i) {
        ASSERT_FLOAT_EQ(r->buf()[i], matrix_c[i]);
    }
}
