// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#include "tensor.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace pluto {
    auto tensor::create(context* const ctx, const std::span<const dim> shape) -> tensor* {
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
}
