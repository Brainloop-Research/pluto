// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

GTEST_TEST(graph, compute_graph) {
    context ctx {};
    pool_ref<tensor> t1 {tensor::create(&ctx, {4, 4})};
    pool_ref<tensor> t2 {tensor::create(&ctx, {4, 4})};
    t1->fill(0.5f);
    t2->fill(1.0f);
    pool_ref<tensor> r {tensor::create(&ctx, {4, 4})};
    r->set_op(opcode::add, t1, t2);
    backends::cpu::cpu_backend cpu {};
    ASSERT_TRUE(cpu.verify(compute_ctx {}, r, graph_eval_order::left_to_right));
    cpu.compute(compute_ctx {}, r, graph_eval_order::left_to_right);
    std::cout << *r;
}
