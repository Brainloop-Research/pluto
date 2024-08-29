// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "core.hpp"
#include "graph.hpp"

#include <cassert>
#include <numeric>
#include <iosfwd>
#include <span>

namespace pluto {
    using dim = std::int64_t;
    static constexpr dim max_dims {4};
    using multi_dim = std::array<dim, max_dims>;

    template <typename T, typename... Ts>
    concept is_any_of = std::disjunction_v<std::is_same<T, Ts>...>;

    template <typename T>
    concept is_dtype = is_any_of<T, float>;

    class tensor final {
    public:
        static constexpr dim buf_align {alignof(float)};

        tensor() = default;
        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

        [[nodiscard]] static auto create(context* ctx, std::span<const dim> shape) noexcept -> tensor*;
        [[nodiscard]] static auto create(context* ctx, std::initializer_list<const dim> dims) noexcept -> tensor*;
        [[nodiscard]] auto isomorphic_clone() const -> tensor*;
        [[nodiscard]] auto deep_clone() const -> tensor*;
        [[nodiscard]] auto ctx() const noexcept -> context*;
        [[nodiscard]] auto rank() const noexcept -> dim;
        [[nodiscard]] auto shape() const noexcept -> const std::array<dim, max_dims>&;
        [[nodiscard]] auto strides() const noexcept -> const std::array<dim, max_dims>&;
        [[nodiscard]] auto buf() const noexcept -> std::span<float>;
        [[nodiscard]] auto row_count() const noexcept -> dim;
        [[nodiscard]] auto col_count() const noexcept -> dim;
        auto linear_to_multidim_idx(dim i, multi_dim& o) noexcept -> void;
        [[nodiscard]] auto multidim_to_linear_idx(const multi_dim& i) const noexcept -> dim;
        [[nodiscard]] auto is_scalar() const noexcept -> bool;
        [[nodiscard]] auto is_vector() const noexcept -> bool;
        [[nodiscard]] auto is_matrix() const noexcept -> bool;
        [[nodiscard]] auto is_higher_order3d() const noexcept -> bool;
        [[nodiscard]] auto is_shape_eq(const tensor* other) const noexcept -> bool;
        [[nodiscard]] auto is_matmul_compatible(const tensor* other) const noexcept -> bool;

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
        std::array<dim, max_dims> m_shape {}; // Cardinality of each dimension
        std::array<dim, max_dims> m_strides {}; // Byte strides for each dimension
        dim m_rank {}; // Number of dimensions
        std::array<tensor*, max_args> m_args {}; // Arguments for the operation
        std::size_t m_num_args {}; // Number of arguments
        opcode m_op {}; // Operation code

        friend auto operator << (std::ostream&, const tensor&) -> std::ostream&;
    };

    auto operator << (std::ostream& o, const tensor& self) -> std::ostream&;
}
