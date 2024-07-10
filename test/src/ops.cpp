// (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

GTEST_TEST(ops, range_check) {
    constexpr auto min {PT_OPC_NOP};
    constexpr auto max {PT_OPC_MAX};
    ASSERT_EQ(0, min);
    ASSERT_EQ(max-1, PT_OPC_MATMUL);
}