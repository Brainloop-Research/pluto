// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "../../backend.hpp"

namespace pluto::backends::cpu {
    class cpu_backend final : public backend_interface {
    public:
        cpu_backend();
        ~cpu_backend() override = default;

    protected:
        [[nodiscard]] virtual auto eval_softmax (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_sigmoid (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_tanh    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_relu    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_gelu    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_silu    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_add     (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_sub     (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_mul     (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_div     (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
        [[nodiscard]] virtual auto eval_matmul  (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* override;
    };
}
