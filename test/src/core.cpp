// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

GTEST_TEST(core, ctx_init_free) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 0);
    ASSERT_NE(ctx.chunk_size, 0);
    ASSERT_NE(ctx.chunks, nullptr);
    ASSERT_EQ(ctx.chunks_len, 1);
    pt_ctx_free(&ctx);
}

GTEST_TEST(core, ctx_pool_exhaust_chunk) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, nullptr, 1);
    for (int i {1}; i < 1000; ++i) {
        int *const x = static_cast<int *>(pt_ctx_pool_alloc(&ctx, sizeof(int)*i));
        *x = i;
        ASSERT_EQ(*x, i);
    }
    pt_ctx_free(&ctx);
}

GTEST_TEST(core, hpc_clock) {
    std::uint64_t prev = pt_hpc_micro_clock();
    for (int i {0}; i < 1000; ++i) {
        std::uint64_t now = pt_hpc_micro_clock();
        ASSERT_NE(0, now);
        ASSERT_LE(prev, now);
        prev = now;
    }
}
