// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "cpu_backend.hpp"
#include "blas.hpp"

namespace pluto::backends::cpu {
    cpu_backend::cpu_backend() : backend_interface {"cpu"} {

    }

    auto cpu_backend::eval_softmax(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        blas::t_softmax(ctx, *node, *node->get_args()[0]);
    }

    auto cpu_backend::eval_sigmoid(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_sigmoid(ctx, *node, *node->get_args()[0]);
    }

    auto cpu_backend::eval_tanh(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_tanh(ctx, *node, *node->get_args()[0]);
    }

    auto cpu_backend::eval_relu(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_relu(ctx, *node, *node->get_args()[0]);
    }

    auto cpu_backend::eval_gelu(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_gelu(ctx, *node, *node->get_args()[0]);
    }

    auto cpu_backend::eval_silu(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_silu(ctx, *node, *node->get_args()[0]);
    }

    auto cpu_backend::eval_add(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_add(ctx, *node, *node->get_args()[0], *node->get_args()[1]);
    }

    auto cpu_backend::eval_sub(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_sub(ctx, *node, *node->get_args()[0], *node->get_args()[1]);
    }

    auto cpu_backend::eval_mul(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_mul(ctx, *node, *node->get_args()[0], *node->get_args()[1]);
    }

    auto cpu_backend::eval_div(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_div(ctx, *node, *node->get_args()[0], *node->get_args()[1]);
    }

    auto cpu_backend::eval_matmul(const compute_ctx& ctx, tensor* const node) const noexcept -> void {
        return blas::t_matmul(ctx, *node, *node->get_args()[0], *node->get_args()[1]);
    }
}
