// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(tensor, tensor_new_1d) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    auto *t = pt_tensor_new_1d(&ctx, 10);
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank, 1);
    ASSERT_EQ(t->shape[0], 10);
    ASSERT_EQ(t->shape[1], 1);
    ASSERT_EQ(t->shape[2], 1);
    ASSERT_EQ(t->shape[3], 1);
    ASSERT_EQ(t->size, 10*sizeof(float));
    ASSERT_EQ(pt_tensor_num_elems(t), 10);
    ASSERT_EQ(t->strides[0], sizeof(float));
    ASSERT_EQ(t->strides[1], 10*sizeof(float));
    ASSERT_EQ(t->strides[2], 10*sizeof(float));
    ASSERT_EQ(t->strides[3], 10*sizeof(float));
    pt_ctx_free(&ctx);
}

TEST(tensor, tensor_new_2d) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    auto *t = pt_tensor_new_2d(&ctx, 4, 4);
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank, 2);
    ASSERT_EQ(t->shape[0], 4);
    ASSERT_EQ(t->shape[1], 4);
    ASSERT_EQ(t->shape[2], 1);
    ASSERT_EQ(t->shape[3], 1);
    ASSERT_EQ(t->size, 4*4*sizeof(float));
    ASSERT_EQ(pt_tensor_num_elems(t), 4*4);
    ASSERT_EQ(t->strides[0], sizeof(float));
    ASSERT_EQ(t->strides[1], 4*sizeof(float));
    ASSERT_EQ(t->strides[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides[3], 4*4*sizeof(float));
    pt_ctx_free(&ctx);
}

TEST(tensor, tensor_new_3d) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    auto *t = pt_tensor_new_3d(&ctx, 4, 4, 8);
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank, 3);
    ASSERT_EQ(t->shape[0], 4);
    ASSERT_EQ(t->shape[1], 4);
    ASSERT_EQ(t->shape[2], 8);
    ASSERT_EQ(t->shape[3], 1);
    ASSERT_EQ(t->size, 4*4*8*sizeof(float));
    ASSERT_EQ(pt_tensor_num_elems(t), 4*4*8);
    ASSERT_EQ(t->strides[0], sizeof(float));
    ASSERT_EQ(t->strides[1], 4*sizeof(float));
    ASSERT_EQ(t->strides[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides[3], 4*4*8*sizeof(float));
    pt_ctx_free(&ctx);
}

TEST(tensor, tensor_new_4d) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    auto *t = pt_tensor_new_4d(&ctx, 4, 4, 8, 2);
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank, 4);
    ASSERT_EQ(t->shape[0], 4);
    ASSERT_EQ(t->shape[1], 4);
    ASSERT_EQ(t->shape[2], 8);
    ASSERT_EQ(t->shape[3], 2);
    ASSERT_EQ(t->size, 4*4*8*2*sizeof(float));
    ASSERT_EQ(pt_tensor_num_elems(t), 4*4*8*2);
    ASSERT_EQ(t->strides[0], sizeof(float));
    ASSERT_EQ(t->strides[1], 4*sizeof(float));
    ASSERT_EQ(t->strides[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides[3], 4*4*8*sizeof(float));
    pt_ctx_free(&ctx);
}

TEST(tensor, tensor_fill) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    auto *origin = pt_tensor_new_4d(&ctx, 4, 4, 8, 2);
    pt_tensor_fill(origin, -0.5f);
    for (pt_dim_t i = 0; i < origin->size / sizeof(float); ++i) {
        ASSERT_FLOAT_EQ(origin->data[i], -0.5f);
    }
    pt_ctx_free(&ctx);
}

static float fill_fn(pt_dim_t) {
    return 1.5f;
}

TEST(tensor, tensor_fill_fn) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    auto *t = pt_tensor_new_4d(&ctx, 4, 4, 8, 2);
    pt_tensor_fill_fn(t, &fill_fn);
    for (pt_dim_t i = 0; i < pt_tensor_num_elems(t); ++i) {
        ASSERT_FLOAT_EQ(t->data[i], 1.5f);
    }
    pt_ctx_free(&ctx);
}

TEST(tensor, tensor_isomorphic) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    auto *origin = pt_tensor_new_4d(&ctx, 4, 4, 8, 2);
    pt_tensor_fill(origin, -0.5f);
    auto *t = pt_tensor_isomorphic(&ctx, origin);
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank, 4);
    ASSERT_EQ(t->shape[0], 4);
    ASSERT_EQ(t->shape[1], 4);
    ASSERT_EQ(t->shape[2], 8);
    ASSERT_EQ(t->shape[3], 2);
    ASSERT_EQ(t->size, 4*4*8*2*sizeof(float));
    ASSERT_EQ(pt_tensor_num_elems(t), 4*4*8*2);
    ASSERT_EQ(t->strides[0], sizeof(float));
    ASSERT_EQ(t->strides[1], 4*sizeof(float));
    ASSERT_EQ(t->strides[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides[3], 4*4*8*sizeof(float));
    for (pt_dim_t i = 0; i < pt_tensor_num_elems(t); ++i) {
        ASSERT_NE(t->data[i], -0.5f);
    }
    pt_ctx_free(&ctx);
}

TEST(tensor, tensor_clone) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    auto *origin = pt_tensor_new_4d(&ctx, 4, 4, 8, 2);
    pt_tensor_fill(origin, -0.5f);
    auto *t = pt_tensor_clone(&ctx, origin);
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->rank, 4);
    ASSERT_EQ(t->shape[0], 4);
    ASSERT_EQ(t->shape[1], 4);
    ASSERT_EQ(t->shape[2], 8);
    ASSERT_EQ(t->shape[3], 2);
    ASSERT_EQ(t->size, 4*4*8*2*sizeof(float));
    ASSERT_EQ(pt_tensor_num_elems(t), 4*4*8*2);
    ASSERT_EQ(t->strides[0], sizeof(float));
    ASSERT_EQ(t->strides[1], 4*sizeof(float));
    ASSERT_EQ(t->strides[2], 4*4*sizeof(float));
    ASSERT_EQ(t->strides[3], 4*4*8*sizeof(float));
    for (pt_dim_t i = 0; i < pt_tensor_num_elems(t); ++i) {
        ASSERT_FLOAT_EQ(t->data[i], -0.5f);
    }
    pt_ctx_free(&ctx);
}
