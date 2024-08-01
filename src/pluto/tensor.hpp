// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "core.hpp"
#include "graph.hpp"

#include <cassert>
#include <numeric>
#include <span>

namespace pluto {
    using dim = std::int64_t;
    static constexpr dim max_dims {4};
    using multi_dim = std::array<dim, max_dims>;

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
        [[nodiscard]] static inline auto create(context* ctx, const std::initializer_list<const dim> dims) noexcept -> tensor* {
            return create(ctx, std::span<const dim>{dims});
        }
        [[nodiscard]] auto isomorphic_clone() const -> tensor* {
            return create(m_ctx, {m_shape.begin(), m_shape.begin()+m_rank});
        }
        [[nodiscard]] auto deep_clone() const -> tensor* {
            auto* const t {this->isomorphic_clone()};
            std::copy(m_buf.begin(), m_buf.end(), t->m_buf.begin());
            return t;
        }

        [[nodiscard]] auto ctx() const noexcept -> context* { return m_ctx; }
        [[nodiscard]] auto rank() const noexcept -> dim { return m_rank; }
        [[nodiscard]] auto shape() const noexcept -> const std::array<dim, max_dims>& { return m_shape; }
        [[nodiscard]] auto strides() const noexcept -> const std::array<dim, max_dims>& { return m_strides; }
        [[nodiscard]] auto buf() const noexcept -> std::span<float> { return m_buf; }
        [[nodiscard]] auto row_count() const noexcept -> dim  {
            return std::accumulate(m_shape.begin()+1, m_shape.end(), 1, std::multiplies<>{});
        }
        [[nodiscard]] auto col_count() const noexcept -> dim {
            return m_shape[0];
        }
        auto linear_to_multidim_idx(const dim i, multi_dim& o) noexcept -> void {
            const auto [d0, d1, d2, _] {m_shape};
            o[3] = i / (d2*d1*d0);
            o[2] = (i - o[3]*d2*d1*d0) / (d1*d0);
            o[1] = (i - o[3]*d2*d1*d0 - o[2]*d1*d0) / d0;
            o[0] =  i - o[3]*d2*d1*d0 - o[2]*d1*d0 - o[1]*d0;
        }
        [[nodiscard]] auto multidim_to_linear_idx(const multi_dim& i) const noexcept -> dim {
           return static_cast<dim>(std::inner_product(i.begin(), i.end(), m_strides.begin(), dim{}) / sizeof(float));
        }
        [[nodiscard]] auto is_scalar() const noexcept -> bool {
            return std::all_of(m_shape.begin(), m_shape.end(), [](const dim d) noexcept -> bool { return d == 1; });
        }
        [[nodiscard]] auto is_vector() const noexcept -> bool {
            return std::all_of(m_shape.begin()+1, m_shape.end(), [](const dim d) noexcept -> bool { return d == 1; });
        }
        [[nodiscard]] auto is_matrix() const noexcept -> bool {
            return std::all_of(m_shape.begin()+2, m_shape.end(), [](const dim d) noexcept -> bool { return d == 1; });
        }
        [[nodiscard]] auto is_higher_order3d() const noexcept -> bool { return m_shape[max_dims-1] == 1; }
        [[nodiscard]] auto is_shape_eq(const tensor* const other) const noexcept -> bool { return m_rank == other->m_rank && m_shape == other->m_shape; }
        [[nodiscard]] auto is_matmul_compatible(const tensor* const other) const noexcept -> bool { return m_shape[0] == other->m_shape[1]; }

        auto fill(const float val) noexcept -> void {
            if (val == 0.0f) std::memset(m_buf.data(), 0, m_buf.size()*sizeof(float));
            else std::fill(m_buf.begin(), m_buf.end(), val);
        }
        auto populate(const std::span<const float> values) noexcept -> void {
            assert(m_buf.size() == values.size());
            std::copy(values.begin(), values.end(), m_buf.begin());
        }

        template <typename F> requires std::is_invocable_r_v<float, F, dim>
        auto fill_fn(F&& f) noexcept(std::is_nothrow_invocable_r_v<float, F, dim>) -> void {
            const auto n {static_cast<dim>(m_buf.size())};
            for (dim i {}; i < n; ++i) {
                m_buf[i] = std::invoke(f, i);
            }
        }

        [[nodiscard]] auto get_args() noexcept -> std::span<tensor*> { return {m_args.data(), m_num_args}; }
        [[nodiscard]] auto get_op_code() const noexcept -> graph::opcode { return m_op; }

    private:
        context* m_ctx {}; // Context host
        std::span<float> m_buf {}; // Pointer to the data
        std::array<dim, max_dims> m_shape {}; // Cardinality of each dimension
        std::array<dim, max_dims> m_strides {}; // Byte strides for each dimension
        dim m_rank {}; // Number of dimensions
        [[maybe_unused]] std::array<tensor*, graph::max_args> m_args {}; // Arguments for the operation
        std::size_t m_num_args {}; // Number of arguments
        [[maybe_unused]] graph::opcode m_op {}; // Operation code
    };
}
