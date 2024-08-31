// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(tensor, tensor_new_1d) {
    context ctx {};
    auto* t {tensor::create(&ctx, {10})};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape().rank(), 1);
    ASSERT_EQ(t->shape()[0], 10);
    ASSERT_EQ(t->shape()[1], 1);
    ASSERT_EQ(t->shape()[2], 1);
    ASSERT_EQ(t->shape()[3], 1);
    ASSERT_EQ(t->buf().size(), 10);
    ASSERT_EQ(t->shape().colums(), 10);
    ASSERT_EQ(t->shape().rows(), 1);
    ASSERT_EQ(t->shape().strides()[0], sizeof(float));
    ASSERT_EQ(t->shape().strides()[1], 10*sizeof(float));
    ASSERT_EQ(t->shape().strides()[2], 10*sizeof(float));
    ASSERT_EQ(t->shape().strides()[3], 10*sizeof(float));
    multi_dim idx {t->shape().to_multi_dim_index(5)};
    ASSERT_EQ(t->shape().to_linear_index(idx), 5);
    ASSERT_EQ(idx[0], 5);
    ASSERT_EQ(idx[1], 0);
    ASSERT_EQ(idx[2], 0);
    ASSERT_EQ(idx[3], 0);
    ASSERT_FALSE(t->shape().is_scalar());
    ASSERT_TRUE(t->shape().is_vector());
    ASSERT_TRUE(t->shape().is_matrix());
    ASSERT_TRUE(t->shape().is_higher_order3d());
}

TEST(tensor, tensor_new_2d) {
    context ctx {};
    auto* t {tensor::create(&ctx, {4, 4})};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape().rank(), 2);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 1);
    ASSERT_EQ(t->shape()[3], 1);
    ASSERT_EQ(t->buf().size(), 4 * 4);
    ASSERT_EQ(t->shape().colums(), 4);
    ASSERT_EQ(t->shape().rows(), 4);
    ASSERT_EQ(t->shape().strides()[0], sizeof(float));
    ASSERT_EQ(t->shape().strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[3], 4*4*sizeof(float));
    multi_dim idx {t->shape().to_multi_dim_index(5)};
    ASSERT_EQ(t->shape().to_linear_index(idx), 5);
    ASSERT_EQ(idx[0], 1);
    ASSERT_EQ(idx[1], 1);
    ASSERT_EQ(idx[2], 0);
    ASSERT_EQ(idx[3], 0);
    ASSERT_FALSE(t->shape().is_scalar());
    ASSERT_FALSE(t->shape().is_vector());
    ASSERT_TRUE(t->shape().is_matrix());
    ASSERT_TRUE(t->shape().is_higher_order3d());
}

TEST(tensor, tensor_new_3d) {
    context ctx {};
    auto* t {tensor::create(&ctx, {4, 4, 8})};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape().rank(), 3);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 8);
    ASSERT_EQ(t->shape()[3], 1);
    ASSERT_EQ(t->buf().size(), 4 * 4 * 8);
    ASSERT_EQ(t->shape().colums(), 4);
    ASSERT_EQ(t->shape().rows(), 4*8);
    ASSERT_EQ(t->shape().strides()[0], sizeof(float));
    ASSERT_EQ(t->shape().strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[3], 4*4*8*sizeof(float));
    multi_dim idx {t->shape().to_multi_dim_index(13)};
    ASSERT_EQ(t->shape().to_linear_index(idx), 13);
    ASSERT_EQ(idx[0], 1);
    ASSERT_EQ(idx[1], 3);
    ASSERT_EQ(idx[2], 0);
    ASSERT_EQ(idx[3], 0);
    ASSERT_FALSE(t->shape().is_scalar());
    ASSERT_FALSE(t->shape().is_vector());
    ASSERT_FALSE(t->shape().is_matrix());
    ASSERT_TRUE(t->shape().is_higher_order3d());
}

TEST(tensor, tensor_new_4d) {
    context ctx {};
    auto* t {tensor::create(&ctx, {4, 4, 8, 2})};
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape().rank(), 4);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 8);
    ASSERT_EQ(t->shape()[3], 2);
    ASSERT_EQ(t->buf().size(), 4 * 4 * 8 * 2);
    ASSERT_EQ(t->shape().colums(), 4);
    ASSERT_EQ(t->shape().rows(), 4*8*2);
    ASSERT_EQ(t->shape().strides()[0], sizeof(float));
    ASSERT_EQ(t->shape().strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[3], 4*4*8*sizeof(float));
    multi_dim idx {t->shape().to_multi_dim_index(28)};
    ASSERT_EQ(t->shape().to_linear_index(idx), 28);
    ASSERT_EQ(idx[0], 0);
    ASSERT_EQ(idx[1], 3);
    ASSERT_EQ(idx[2], 1);
    ASSERT_EQ(idx[3], 0);
    ASSERT_FALSE(t->shape().is_scalar());
    ASSERT_FALSE(t->shape().is_vector());
    ASSERT_FALSE(t->shape().is_matrix());
    ASSERT_FALSE(t->shape().is_higher_order3d());
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
    ASSERT_EQ(t->shape().rank(), 4);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 8);
    ASSERT_EQ(t->shape()[3], 2);
    ASSERT_EQ(t->buf().size(), 4 * 4 * 8 * 2);
    ASSERT_EQ(t->shape().strides()[0], sizeof(float));
    ASSERT_EQ(t->shape().strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[3], 4*4*8*sizeof(float));
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
    ASSERT_EQ(t->shape().rank(), 4);
    ASSERT_EQ(t->shape()[0], 4);
    ASSERT_EQ(t->shape()[1], 4);
    ASSERT_EQ(t->shape()[2], 8);
    ASSERT_EQ(t->shape()[3], 2);
    ASSERT_EQ(t->buf().size(), 4 * 4 * 8 * 2);
    ASSERT_EQ(t->shape().strides()[0], sizeof(float));
    ASSERT_EQ(t->shape().strides()[1], 4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(t->shape().strides()[3], 4*4*8*sizeof(float));
    for (const float x : t->buf()) {
        ASSERT_FLOAT_EQ(x, -0.5f);
    }
}
