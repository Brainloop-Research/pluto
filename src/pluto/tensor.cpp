// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "tensor.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace pluto {
    auto tensor::create(context* const ctx, const std::span<const dim> shape) noexcept -> tensor* {
        assert(ctx != nullptr);
        auto* const t {ctx->pool_alloc<tensor>()}; // Allocate memory for the tensor
        t->m_ctx = ctx;
        constexpr dim scalar_size {sizeof(float)};
        dim size {static_cast<dim>(std::accumulate(shape.begin(), shape.end(), scalar_size, std::multiplies<>{}))};
        auto* const buf {static_cast<float*>(ctx->pool_alloc_raw_aligned(size, buf_align))};
        t->m_buf = {buf, static_cast<std::size_t>(size / scalar_size)}; // Allocate memory for the data
        std::fill(t->m_buf.begin(), t->m_buf.end(), 0); // Zero-initialize the data TODO: remove
        std::fill(t->m_shape.begin(), t->m_shape.end(), 1); // Set dimensions and strides to identity to saturate out zero multiplication because: x * 0 = 0
        std::copy(shape.begin(), shape.end(), t->m_shape.begin());
        t->m_strides[0] = scalar_size;
        for (dim i {1}; i < max_dims; ++i)
            t->m_strides[i] = t->m_strides[i-1]*t->m_shape[i-1];
        t->m_rank = static_cast<dim>(shape.size());
        return t;
    }

    auto tensor::create(context* ctx, const std::initializer_list<const dim> dims) noexcept -> tensor* {
        return create(ctx, std::span<const dim>{dims});
    }

    auto tensor::isomorphic_clone() const -> tensor* {
        return create(m_ctx, {m_shape.begin(), m_shape.begin()+m_rank});
    }

    auto tensor::deep_clone() const -> tensor* {
        auto* const t {this->isomorphic_clone()};
        std::copy(m_buf.begin(), m_buf.end(), t->m_buf.begin());
        return t;
    }

    auto tensor::ctx() const noexcept -> context* { return m_ctx; }

    auto tensor::rank() const noexcept -> dim { return m_rank; }

    auto tensor::shape() const noexcept -> const std::array<dim, max_dims>& { return m_shape; }

    auto tensor::strides() const noexcept -> const std::array<dim, max_dims>& { return m_strides; }

    auto tensor::buf() const noexcept -> std::span<float> { return m_buf; }

    auto tensor::row_count() const noexcept -> dim  {
        return std::accumulate(m_shape.begin()+1, m_shape.end(), 1, std::multiplies<>{});
    }

    auto tensor::col_count() const noexcept -> dim {
        return m_shape[0];
    }

    auto tensor::linear_to_multidim_idx(const dim i, multi_dim& o) noexcept -> void {
        const auto [d0, d1, d2, _] {m_shape};
        o[3] = i / (d2*d1*d0);
        o[2] = (i - o[3]*d2*d1*d0) / (d1*d0);
        o[1] = (i - o[3]*d2*d1*d0 - o[2]*d1*d0) / d0;
        o[0] =  i - o[3]*d2*d1*d0 - o[2]*d1*d0 - o[1]*d0;
    }

    auto tensor::multidim_to_linear_idx(const multi_dim& i) const noexcept -> dim {
        return static_cast<dim>(std::inner_product(i.begin(), i.end(), m_strides.begin(), dim{}) / sizeof(float));
    }

    auto tensor::is_scalar() const noexcept -> bool {
        return std::all_of(m_shape.begin(), m_shape.end(), [](const dim d) noexcept -> bool { return d == 1; });
    }

    auto tensor::is_vector() const noexcept -> bool {
        return std::all_of(m_shape.begin()+1, m_shape.end(), [](const dim d) noexcept -> bool { return d == 1; });
    }

    auto tensor::is_matrix() const noexcept -> bool {
        return std::all_of(m_shape.begin()+2, m_shape.end(), [](const dim d) noexcept -> bool { return d == 1; });
    }

    auto tensor::is_higher_order3d() const noexcept -> bool {
        return m_shape[max_dims-1] == 1;
    }

    auto tensor::is_shape_eq(const tensor* const other) const noexcept -> bool {
        return m_rank == other->m_rank && m_shape == other->m_shape;
    }

    auto tensor::is_matmul_compatible(const tensor* const other) const noexcept -> bool {
        return m_shape[0] == other->m_shape[1];
    }

    auto tensor::fill(const float val) noexcept -> void {
        std::fill(m_buf.begin(), m_buf.end(), val);
    }

    auto tensor::populate(const std::span<const float> values) noexcept -> void {
        assert(m_buf.size() == values.size());
        std::copy(values.begin(), values.end(), m_buf.begin());
    }

    auto tensor::get_args() const noexcept -> std::span<const tensor*> { return {const_cast<const tensor**>(m_args.data()), m_num_args}; }

    auto tensor::get_op_code() const noexcept -> opcode { return m_op; }
}
