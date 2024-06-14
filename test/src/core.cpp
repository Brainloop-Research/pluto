// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

GTEST_TEST(core, ctx_allocate) {
    pt_ctx_t ctx {};
    pt_ctx_init(&ctx, NULL, 0);
    ASSERT_NE(ctx.chunk_size, 0);
    void *const p = pt_ctx_pool_alloc(&ctx, 1<<20);
    ASSERT_NE(p, nullptr);
    pt_ctx_free(&ctx);
}
