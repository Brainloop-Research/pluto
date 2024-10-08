// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <numeric>

#include "prelude.hpp"
#include "pluto/backends/cpu/blas.hpp"

#include <pluto/backends/cpu/blas_impl.inl>

using namespace pluto;
using namespace backends::cpu::blas;

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
    v_cvt_f16_to_f32(f16_vec.size(), f32_out.data(), f16_vec.data());
    for (std::size_t i {0}; i < f32_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(f32_out[i] - f32_vec[i])
            < static_cast<float>(f16::eps())); // |ξ1 - ξ2| < ε
    }
}

GTEST_TEST(blas, cvt_f32_to_f16_vec) {
    std::array<f16, 16> f16_out {};
    v_cvt_f32_to_f16(f32_vec.size(), f16_out.data(), f32_vec.data());
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
    v_cvt_bf16_to_f32(bf16_vec.size(), f32_out.data(), bf16_vec.data());
    for (std::size_t i {0}; i < f32_vec.size(); ++i) {
        ASSERT_TRUE(std::abs(f32_out[i] - f32_vec[i])
                    < static_cast<float>(bf16::eps())); // |ξ1 - ξ2| < ε
    }
}

GTEST_TEST(blas, cvt_f32_to_bf16_vec) {
    std::array<bf16, 16> bf16_out {};
    v_cvt_f32_to_bf16(f32_vec.size(), bf16_out.data(), f32_vec.data());
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
    v_softmax(data.size(), r.data(), data.data());
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
    v_sigmoid(data.size(), r.data(), data.data());
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
    v_tanh(data.size(), r.data(), data.data());
    for (std::size_t i {}; i < data.size(); ++i) {
        ASSERT_EQ(r[i], std::tanh(data[i]));
    }
}

GTEST_TEST(vblas, relu_f32) {
    std::vector<float> data {};
    data.reserve(325);
    for (std::size_t i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i % 2 == 0 ? -i : i));
    }
    std::vector<float> r {};
    r.resize(data.size());
    v_relu(data.size(), r.data(), data.data());
    for (std::size_t i {}; i < data.size(); ++i) {
        ASSERT_EQ(r[i], std::max(0.0f, data[i]));
    }
}

GTEST_TEST(vblas, gelu_f32) {
    std::vector<float> data {};
    data.reserve(325);
    for (std::size_t i {0}; i < data.capacity(); ++i) {
        data.emplace_back(static_cast<float>(i % 2 == 0 ? -i : i));
    }
    std::vector<float> r {};
    r.resize(data.size());
    v_gelu(data.size(), r.data(), data.data());
    for (std::size_t i {}; i < data.size(); ++i) {
        ASSERT_EQ(r[i], 0.5f * data[i] * (1.0f + std::tanh(sqrt2pi * data[i] * (1.0f + gelu_coeff * data[i] * data[i]))));
    }
}

GTEST_TEST(vblas, add_f32) {
    std::vector<float> x {}, y {}, r {};
    std::generate_n(std::back_inserter(x), 325, []() noexcept -> float { return 1.0f; });
    std::generate_n(std::back_inserter(y), 325, []() noexcept -> float { return 2.0f; });
    r.resize(x.size());
    v_add(x.size(), r.data(), x.data(), y.data());
    for (std::size_t i {}; i < x.size(); ++i) {
        ASSERT_FLOAT_EQ(r[i], x[i] + y[i]);
    }
}

GTEST_TEST(vblas, sub_f32) {
    std::vector<float> x {}, y {}, r {};
    std::generate_n(std::back_inserter(x), 325, []() { return 1.0f; });
    std::generate_n(std::back_inserter(y), 325, []() { return 2.0f; });
    r.resize(x.size());
    v_sub(x.size(), r.data(), x.data(), y.data());
    for (std::size_t i {}; i < x.size(); ++i) {
        ASSERT_FLOAT_EQ(r[i], x[i] - y[i]);
    }
}

GTEST_TEST(vblas, mul_f32) {
    std::vector<float> x {}, y {}, r {};
    std::generate_n(std::back_inserter(x), 325, []() { return 1.0f; });
    std::generate_n(std::back_inserter(y), 325, []() { return 2.0f; });
    r.resize(x.size());
    v_mul(x.size(), r.data(), x.data(), y.data());
    for (std::size_t i {}; i < x.size(); ++i) {
        ASSERT_FLOAT_EQ(r[i], x[i] * y[i]);
    }
}

GTEST_TEST(vblas, div_f32) {
    std::vector<float> x {}, y {}, r {};
    std::generate_n(std::back_inserter(x), 325, []() { return 1.0f; });
    std::generate_n(std::back_inserter(y), 325, []() { return 2.0f; });
    r.resize(x.size());
    v_div(x.size(), r.data(), x.data(), y.data());
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
    const float dot = v_dot(data.size(), data.data(), data.data());
    const float dot_ref = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f);
    ASSERT_FLOAT_EQ(dot, dot_ref);
}

GTEST_TEST(blas, tensor_softmax) {
    constexpr float x1 {0.7f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    v_softmax(1, &r1, &x1);
    t1->fill(x1);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_softmax(compute_ctx{}, *r, *t1);
    ASSERT_TRUE(r->shape() == t1->shape());
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_sigmoid) {
    constexpr float x1 {0.7f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    v_sigmoid(1, &r1, &x1);
    t1->fill(x1);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_sigmoid(compute_ctx{}, *r, *t1);
    ASSERT_TRUE(r->shape() == t1->shape());
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_tanh) {
    constexpr float x1 {0.7f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    v_tanh(1, &r1, &x1);
    t1->fill(x1);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_tanh(compute_ctx{}, *r, *t1);
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_relu) {
    constexpr float x1 {0.7f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    v_relu(1, &r1, &x1);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_relu(compute_ctx{}, *r, *t1);
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_gelu) {
    constexpr float x1 {0.7f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    v_gelu(1, &r1, &x1);
    t1->fill(x1);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_gelu(compute_ctx{}, *r, *t1);
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_silu) {
    constexpr float x1 {0.7f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    float r1 {};
    v_silu(1, &r1, &x1);
    t1->fill(x1);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_silu(compute_ctx{}, *r, *t1);
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, r1);
    }
}

GTEST_TEST(blas, tensor_add_f32) {
    constexpr float x1 {1.0f}, x2 {2.0f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    pool_ref<tensor> t2 {tensor::create(&ctx, {4*4, 4*8, 8*2, 2})};
    t1->fill(x1);
    t2->fill(x2);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_add(compute_ctx{}, *r, *t1, *t2);
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, x1 + x2);
    }
}

GTEST_TEST(blas, tensor_sub_f32) {
    constexpr float x1 {1.0f}, x2 {2.0f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    pool_ref<tensor> t2 {tensor::create(&ctx, {4*4, 4*8, 8*2, 2})};
    t1->fill(x1);
    t2->fill(x2);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_sub(compute_ctx{}, *r, *t1, *t2);
    ASSERT_TRUE(r->shape() == t1->shape());
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, x1 - x2);
    }
}

GTEST_TEST(blas, tensor_mul_f32) {
    constexpr float x1 {1.0f}, x2 {2.0f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    pool_ref<tensor> t2 {tensor::create(&ctx, {4*4, 4*8, 8*2, 2})};
    t1->fill(x1);
    t2->fill(x2);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_mul(compute_ctx{}, *r, *t1, *t2);
    ASSERT_TRUE(r->shape() == t1->shape());
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, x1 * x2);
    }
}

GTEST_TEST(blas, tensor_div_f32) {
    constexpr float x1 {1.0f}, x2 {2.0f};
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4*4, 4*9, 8*2, 2})};
    pool_ref<tensor> t2 {tensor::create(&ctx, {4*4, 4*8, 8*2, 2})};
    t1->fill(x1);
    t2->fill(x2);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_div(compute_ctx{}, *r, *t1, *t2);
    for (const float x : r->buf()) {
        ASSERT_FLOAT_EQ(x, x1 / x2);
    }
}

// matrix A (MxK)
static constexpr std::array<float, 4*4> matrix_a {
    1, 3, 8, 9,
    8, 5, 5, 1,
    1, 6, 3, 5,
    3, 1, 3, 2
};

// matrix B (NxK)
static constexpr std::array<float, 4*4> matrix_b {
    6, 3, 1, 5,
    9, 1, 5, 4,
    9, 6, 5, 5,
    1, 1, 1, 8
};

// matrix C (MxN)
static constexpr std::array<float, 4*4> matrix_c {
    114, 63, 65, 129,
    139, 60, 59, 93,
    92, 32, 51, 84,
    56, 30, 25, 50
};

GTEST_TEST(blas, tensor_sgemm_f32) {
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4, 4})};
    pool_ref<tensor> t2 {tensor::create(&ctx, {4, 4})};
    t1->populate(matrix_a);
    t2->populate(matrix_b);
    pool_ref<tensor> r {t1->isomorphic_clone()};
    t_matmul(compute_ctx{}, *r, *t1, *t2);
    for (std::size_t i {}; i < matrix_c.size(); ++i) {
        ASSERT_FLOAT_EQ(r->buf()[i], matrix_c[i]);
    }
}
