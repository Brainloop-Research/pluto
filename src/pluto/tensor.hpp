// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "core.hpp"

#include <numeric>
#include <span>

namespace pluto {
    using dim = std::int64_t;

    class tensor final {
    public:
        static constexpr dim max_dims {4};
        static constexpr dim buf_align {alignof(float)};

        tensor() = default;
        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

        [[nodiscard]] static auto create(context* ctx, std::span<const dim> shape) -> tensor*;
        [[nodiscard]] static inline auto create(context* ctx, const std::initializer_list<const dim> dims) -> tensor* {
            return create(ctx, std::span<const dim>{dims});
        }

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
        auto linear_to_multidim_idx(const dim i, std::array<dim, max_dims>& o) noexcept -> void {
            const auto [d0, d1, d2, _] {m_shape};
            o[3] = i / (d2*d1*d0);
            o[2] = (i - o[3]*d2*d1*d0) / (d1*d0);
            o[1] = (i - o[3]*d2*d1*d0 - o[2]*d1*d0) / d0;
            o[0] =  i - o[3]*d2*d1*d0 - o[2]*d1*d0 - o[1]*d0;
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

    private:
        context* m_ctx {}; // Context host
        std::span<float> m_buf {}; // Pointer to the data
        std::array<dim, max_dims> m_shape {}; // Cardinality of each dimension
        std::array<dim, max_dims> m_strides {}; // Byte strides for each dimension
        dim m_rank {}; // Number of dimensions
    };
}
