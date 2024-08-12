// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <span>

#include "graph.hpp"

namespace pluto {
    class tensor;

    // Context for compute operations
    struct compute_ctx final {
        const std::int64_t thread_idx;     // Current thread index - Must be >= 0
        const std::int64_t num_threads;    // Total number of threads Must be > 0
        constexpr explicit compute_ctx(const std::int64_t thread_idx = 0, const std::int64_t num_threads = 1) noexcept
            : thread_idx{std::max<std::int64_t>(0, thread_idx)}, num_threads{std::max<std::int64_t>(1, num_threads)} {}
    };

    class backend_interface {
    public:
        virtual ~backend_interface() = default;

        [[nodiscard]] inline auto id() const noexcept -> std::uint32_t { return m_id; }
        [[nodiscard]] inline auto name() const noexcept -> const std::string& { return m_name; }

        using verify_routine = auto (backend_interface::*)(const compute_ctx& ctx, std::span<const tensor*> args) const -> bool;
        using eval_routine = auto (backend_interface::*)(const compute_ctx& ctx, std::span<const tensor*> args) const -> tensor*;

        [[nodiscard]] auto verify(const compute_ctx& ctx, tensor* root, graph_eval_order order) -> bool;
        [[nodiscard]] auto compute(const compute_ctx& ctx, tensor* root, graph_eval_order order) -> tensor*;

    protected:
        explicit backend_interface(std::string&& name);

        [[nodiscard]] virtual auto verify_softmax(const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_sigmoid(const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_tanh   (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_relu   (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_gelu   (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_silu   (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_add    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_sub    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_mul    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_div    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;
        [[nodiscard]] virtual auto verify_matmul (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool;

        [[nodiscard]] virtual auto eval_softmax (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_sigmoid (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_tanh    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_relu    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_gelu    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_silu    (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_add     (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_sub     (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_mul     (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_div     (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto eval_matmul  (const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> tensor* = 0;

    private:
        const std::uint32_t m_id;
        const std::string m_name;
        const std::array<verify_routine, static_cast<std::size_t>(opcode::len_)> m_verify_dispatch_table;
        const std::array<eval_routine, static_cast<std::size_t>(opcode::len_)> m_eval_dispatch_table;
    };
}
