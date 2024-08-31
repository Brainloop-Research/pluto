// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "core.hpp"
#include "tensor_shape.hpp"
#include "graph.hpp"

#include <cassert>
#include <numeric>
#include <iosfwd>
#include <span>

namespace pluto {
    class tensor final {
    public:
        static constexpr dim buf_align {alignof(float)};

        tensor() = default;
        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

        [[nodiscard]] static auto create(context* ctx, std::span<const dim> dims) noexcept -> tensor*;
        [[nodiscard]] static auto create(context* ctx, std::initializer_list<const dim> dims) noexcept -> tensor*;
        [[nodiscard]] auto isomorphic_clone() const -> tensor*;
        [[nodiscard]] auto deep_clone() const -> tensor*;
        [[nodiscard]] auto ctx() const noexcept -> context* { return m_ctx; }
        [[nodiscard]] auto buf() const noexcept -> std::span<float> { return m_buf; }
        [[nodiscard]] auto shape() noexcept -> struct tensor_shape<float>& { return m_shape; }
        [[nodiscard]] auto shape() const noexcept -> const struct tensor_shape<float>& { return m_shape; }

        auto fill(float val) noexcept -> void;
        auto populate(std::span<const float> values) noexcept -> void;
        auto fill_random(float min = -1.0f, float max = 1.0f) noexcept -> void;
        [[nodiscard]] auto get_args() const noexcept -> std::span<tensor* const>;
        [[nodiscard]] auto get_op_code() const noexcept -> opcode;
        [[nodiscard]] auto is_leaf_node() const noexcept -> bool;
        auto push_arg(tensor* t) -> void;

        template <typename F> requires std::is_invocable_r_v<float, F, dim>
        auto fill_fn(F&& f) noexcept(std::is_nothrow_invocable_r_v<float, F, dim>) -> void {
            const auto n {static_cast<dim>(m_buf.size())};
            for (dim i {}; i < n; ++i) {
                m_buf[i] = std::invoke(f, i);
            }
        }

        template <typename... Args> requires (sizeof...(Args) > 0)
        auto set_op(const opcode op, Args&&... args) noexcept -> void {
            m_op = op;
            for (auto&& arg : std::initializer_list<std::common_type_t<Args...>>{args...})
                push_arg(arg);
        }

    private:
        context* m_ctx {}; // Context host
        std::span<float> m_buf {}; // Pointer to the data
        struct tensor_shape<float> m_shape {}; // Current shape
        std::array<tensor*, max_args> m_args {}; // Arguments for the operation
        std::size_t m_num_args {}; // Number of arguments
        opcode m_op {}; // Operation code

        friend auto operator << (std::ostream&, const tensor&) -> std::ostream&;
    };

    auto operator << (std::ostream& o, const tensor& self) -> std::ostream&;
}
