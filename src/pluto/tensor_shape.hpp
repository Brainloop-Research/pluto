// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "core.hpp"

#include <array>
#include <algorithm>
#include <cassert>
#include <numeric>

namespace pluto {
    template <typename T, typename... Ts>
    concept is_any_of = std::disjunction_v<std::is_same<T, Ts>...>;

    template <typename T>
    concept is_dtype = is_any_of<T, float>;

    using dim = std::int64_t;
    static constexpr dim max_dims {4};
    using multi_dim = std::array<dim, max_dims>;

    template <typename T> requires is_dtype<T>
    struct tensor_shape final {
    public:
        constexpr tensor_shape() noexcept {
            m_rank = 0;
        }
        constexpr explicit tensor_shape(const std::span<const dim> dims) noexcept {
            std::fill(m_dims.begin(), m_dims.end(), 1); // Set dimensions and strides to identity to saturate out zero multiplication because: x * 0 = 0
            std::copy(dims.begin(), dims.end(), m_dims.begin());
            m_strides[0] = sizeof(T);
            for (dim i {1}; i < max_dims; ++i)
                m_strides[i] = m_strides[i-1] * m_dims[i-1];
            m_rank = static_cast<dim>(dims.size());
        }
        tensor_shape(const tensor_shape&) noexcept = default;
        tensor_shape(tensor_shape&&) noexcept = default;
        auto operator = (const tensor_shape&) -> tensor_shape& = default;
        auto operator = (tensor_shape&&) -> tensor_shape& = default;
        ~tensor_shape() = default;

        [[nodiscard]] constexpr auto is_empty() const noexcept -> bool {
            return m_rank == 0;
        }
        [[nodiscard]] constexpr auto rows() const noexcept -> dim {
            return std::accumulate(m_dims.begin() + 1, m_dims.end(), 1, std::multiplies<>{});
        }
        [[nodiscard]] constexpr auto colums() const noexcept -> dim {
            return m_dims[0];
        }
        [[nodiscard]] constexpr auto is_scalar() const noexcept -> bool {
            return std::all_of(m_dims.begin(), m_dims.end(), [](const dim d) noexcept -> bool { return d == 1; });

        }
        [[nodiscard]] constexpr auto is_vector() const noexcept -> bool {
            return std::all_of(m_dims.begin() + 1, m_dims.end(), [](const dim d) noexcept -> bool { return d == 1; });

        }
        [[nodiscard]] constexpr auto is_matrix() const noexcept -> bool {
            return std::all_of(m_dims.begin() + 2, m_dims.end(), [](const dim d) noexcept -> bool { return d == 1; });

        }
        [[nodiscard]] constexpr auto is_higher_order3d() const noexcept -> bool {
            return m_dims[max_dims - 1] == 1;
        }
        [[nodiscard]] constexpr auto is_matmul_compatible(const tensor_shape& other) const noexcept -> bool {
            return m_dims[0] == other.m_dims[1];

        }
        [[nodiscard]] constexpr auto rank() const noexcept -> dim { return m_rank; }
        [[nodiscard]] constexpr auto dims() const noexcept -> const std::array<dim, max_dims>& { return m_dims; }
        [[nodiscard]] constexpr auto strides() const noexcept -> const std::array<dim, max_dims>& { return m_strides; }
        [[nodiscard]] constexpr auto to_linear_index(const multi_dim& i) const noexcept -> dim {
            return static_cast<dim>(std::inner_product(i.begin(), i.end(), m_strides.begin(), dim{}) / sizeof(float));
        }
        [[nodiscard]] constexpr auto to_multi_dim_index(const dim i) const noexcept -> multi_dim {
            const auto [d0, d1, d2, _] {m_dims};
            multi_dim o;
            o[3] = i / (d2*d1*d0);
            o[2] = (i - o[3]*d2*d1*d0) / (d1*d0);
            o[1] = (i - o[3]*d2*d1*d0 - o[2]*d1*d0) / d0;
            o[0] =  i - o[3]*d2*d1*d0 - o[2]*d1*d0 - o[1]*d0;
            return o;
        }

        template <typename S> requires is_dtype<S>
        constexpr auto is_contiguous() const noexcept -> bool {
            return m_strides.front() == sizeof(S);
        }
        constexpr auto operator == (const tensor_shape& other) const noexcept -> bool {
            return m_rank == other.m_rank && m_dims == other.m_dims;
        }
        constexpr auto operator != (const tensor_shape& other) const noexcept -> bool {
            return !(*this == other);
        }
        constexpr explicit operator std::span<const dim>() const noexcept { // Convert to dimension span of used dimensions
            return {m_dims.begin(), m_dims.begin()+m_rank};
        }
        constexpr explicit operator const std::array<dim, max_dims>&() const noexcept { // Convert to dimension span of used dimensions
            return m_dims;
        }
        constexpr auto operator [](const std::size_t i) const noexcept -> dim {
            assert(i < max_dims);
            return m_dims[i];
        }

    private:
        std::array<dim, max_dims> m_dims {}; // Cardinality of each dimension
        std::array<dim, max_dims> m_strides {}; // Byte strides for each dimension
        dim m_rank {}; // Number of dimensions
    };
}
