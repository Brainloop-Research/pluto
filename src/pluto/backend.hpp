// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <algorithm>
#include <cstdint>

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
        [[nodiscard]] virtual auto softmax(const compute_ctx& ctx, const tensor& x) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto sigmoid(const compute_ctx& ctx, const tensor& x) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto tanh(const compute_ctx& ctx, const tensor& x) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto relu(const compute_ctx& ctx, const tensor& x) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto gelu(const compute_ctx& ctx, const tensor& x) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto silu(const compute_ctx& ctx, const tensor& x) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto add(const compute_ctx& ctx, const tensor& x, const tensor& y) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto sub(const compute_ctx& ctx, const tensor& x, const tensor& y) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto mul(const compute_ctx& ctx, const tensor& x, const tensor& y) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto div(const compute_ctx& ctx, const tensor& x, const tensor& y) const noexcept -> tensor* = 0;
        [[nodiscard]] virtual auto matmul(const compute_ctx& ctx, const tensor& x, const tensor& y) const noexcept -> tensor* = 0;

    protected:
        explicit backend_interface(std::string&& name);

    private:
        const std::uint32_t m_id;
        const std::string m_name;
    };
}
