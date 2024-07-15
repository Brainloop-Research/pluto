// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(tensor, tensor_new_1d) {
    context ctx {};
    auto* t {tensor::create(&ctx, {10})};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank(), 1);
    ASSERT_EQ(t->shape()[0], 10);
    ASSERT_EQ(t->shape()[1], 1);
    ASSERT_EQ(t->shape()[2], 1);
    ASSERT_EQ(t->shape()[3], 1);
    ASSERT_EQ(t->buf().size(), 10);
    ASSERT_EQ(t->col_count(), 10);
    ASSERT_EQ(t->row_count(), 1);
    ASSERT_EQ(t->strides()[0], sizeof(float));
    ASSERT_EQ(t->strides()[1], 10*sizeof(float));
    ASSERT_EQ(t->strides()[2], 10*sizeof(float));
    ASSERT_EQ(t->strides()[3], 10*sizeof(float));
    std::array<dim, tensor::max_dims> idx {};
    t->linear_to_multidim_idx(5, idx);
    ASSERT_EQ(idx[0], 5);
    ASSERT_EQ(idx[1], 0);
    ASSERT_EQ(idx[2], 0);
    ASSERT_EQ(idx[3], 0);
    ASSERT_FALSE(t->is_scalar());
    ASSERT_TRUE(t->is_vector());
    ASSERT_TRUE(t->is_matrix());
    ASSERT_TRUE(t->is_higher_order3d());
}

TEST(tensor, tensor_new_2d) {
    context ctx {};
    auto* t {tensor::create(&ctx, {4, 4})};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank(), 2);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 1);
    ASSERT_EQ(t->shape()[3], 1);
    ASSERT_EQ(t->buf().size(), 4 * 4);
    ASSERT_EQ(t->col_count(), 4);
    ASSERT_EQ(t->row_count(), 4);
    ASSERT_EQ(t->strides()[0], sizeof(float));
    ASSERT_EQ(t->strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides()[3], 4*4*sizeof(float));
    std::array<dim, tensor::max_dims> idx {};
    t->linear_to_multidim_idx(5, idx);
    ASSERT_EQ(idx[0], 1);
    ASSERT_EQ(idx[1], 1);
    ASSERT_EQ(idx[2], 0);
    ASSERT_EQ(idx[3], 0);
    ASSERT_FALSE(t->is_scalar());
    ASSERT_FALSE(t->is_vector());
    ASSERT_TRUE(t->is_matrix());
    ASSERT_TRUE(t->is_higher_order3d());
}

TEST(tensor, tensor_new_3d) {
    context ctx {};
    auto* t {tensor::create(&ctx, {4, 4, 8})};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank(), 3);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 8);
    ASSERT_EQ(t->shape()[3], 1);
    ASSERT_EQ(t->buf().size(), 4 * 4 * 8);
    ASSERT_EQ(t->col_count(), 4);
    ASSERT_EQ(t->row_count(), 4*8);
    ASSERT_EQ(t->strides()[0], sizeof(float));
    ASSERT_EQ(t->strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides()[3], 4*4*8*sizeof(float));
    std::array<dim, tensor::max_dims> idx {};
    t->linear_to_multidim_idx(13, idx);
    ASSERT_EQ(idx[0], 1);
    ASSERT_EQ(idx[1], 3);
    ASSERT_EQ(idx[2], 0);
    ASSERT_EQ(idx[3], 0);
    ASSERT_FALSE(t->is_scalar());
    ASSERT_FALSE(t->is_vector());
    ASSERT_FALSE(t->is_matrix());
    ASSERT_TRUE(t->is_higher_order3d());
}

TEST(tensor, tensor_new_4d) {
    context ctx {};
    auto* t {tensor::create(&ctx, {4, 4, 8, 2})};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank(), 4);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 8);
    ASSERT_EQ(t->shape()[3], 2);
    ASSERT_EQ(t->buf().size(), 4 * 4 * 8 * 2);
    ASSERT_EQ(t->col_count(), 4);
    ASSERT_EQ(t->row_count(), 4*8*2);
    ASSERT_EQ(t->strides()[0], sizeof(float));
    ASSERT_EQ(t->strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides()[3], 4*4*8*sizeof(float));
    std::array<dim, tensor::max_dims> idx {};
    t->linear_to_multidim_idx(28, idx);
    ASSERT_EQ(idx[0], 0);
    ASSERT_EQ(idx[1], 3);
    ASSERT_EQ(idx[2], 1);
    ASSERT_EQ(idx[3], 0);
    ASSERT_FALSE(t->is_scalar());
    ASSERT_FALSE(t->is_vector());
    ASSERT_FALSE(t->is_matrix());
    ASSERT_FALSE(t->is_higher_order3d());
}

TEST(tensor, tensor_fill) {
    context ctx {};
    auto* t {tensor::create(&ctx, {4, 4, 8, 2})};
    t->fill(-0.5f);
    for (const float x : t->buf()) {
        ASSERT_FLOAT_EQ(x, -0.5f);
    }
}

TEST(tensor, tensor_fill_fn) {
    context ctx {};
    auto* t {tensor::create(&ctx, {4, 4, 8, 2})};
    t->fill_fn([](dim) { return -1.5f; });
    for (const float x : t->buf()) {
        ASSERT_FLOAT_EQ(x, -1.5f);
    }
}

TEST(tensor, tensor_isomorphic) {
    context ctx {};
    auto* origin {tensor::create(&ctx, {4, 4, 8, 2})};
    origin->fill(-0.5f);
    auto* t {origin->isomorphic_clone()};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank(), 4);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 8);
    ASSERT_EQ(t->shape()[3], 2);
    ASSERT_EQ(t->buf().size(), 4 * 4 * 8 * 2);
    ASSERT_EQ(t->strides()[0], sizeof(float));
    ASSERT_EQ(t->strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides()[3], 4*4*8*sizeof(float));
    for (const float x : t->buf()) {
        ASSERT_NE(x, -0.5f);
    }
}

TEST(tensor, tensor_clone) {
    context ctx {};
    auto* origin {tensor::create(&ctx, {4, 4, 8, 2})};
    origin->fill(-0.5f);
    auto* t {origin->deep_clone()};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank(), 4);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 8);
    ASSERT_EQ(t->shape()[3], 2);
    ASSERT_EQ(t->buf().size(), 4 * 4 * 8 * 2);
    ASSERT_EQ(t->strides()[0], sizeof(float));
    ASSERT_EQ(t->strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides()[3], 4*4*8*sizeof(float));
    for (const float x : t->buf()) {
        ASSERT_FLOAT_EQ(x, -0.5f);
    }
}
