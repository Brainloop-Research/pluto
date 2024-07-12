// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

#include <bit>

GTEST_TEST(core, ctx_init_free) {
    context ctx {};
}

GTEST_TEST(core, ctx_pool_alloc) {
    context ctx {1, 1};
    int *const x = static_cast<int *>(ctx.pool_alloc_raw(sizeof(int)));
    *x = 42;
    ASSERT_EQ(*x, 42);
}

GTEST_TEST(core, ctx_pool_alloc_aligned) {
    context ctx {};
    int *const x = static_cast<int *>(ctx.pool_alloc_raw_aligned(sizeof(int), 64));
    *x = 42;
    ASSERT_EQ(*x, 42);
    ASSERT_EQ(0, std::bit_cast<std::uintptr_t>(x) % 64);
    for (int i {1}; i < 1000; i <<= 1) {
        int *const x = static_cast<int *>(ctx.pool_alloc_raw_aligned(sizeof(int), i));
        ASSERT_EQ(0, std::bit_cast<std::uintptr_t>(x) % i);
    }
}

GTEST_TEST(core, ctx_pool_exhaust_chunk) {
    context ctx {1, 1};
    for (int i {1}; i < 1000; ++i) {
        int *const x = static_cast<int *>(ctx.pool_alloc_raw(sizeof(int)*i));
        *x = i;
        ASSERT_EQ(*x, i);
    }
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
