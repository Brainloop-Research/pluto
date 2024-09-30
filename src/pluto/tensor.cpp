// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "tensor.hpp"

#include <algorithm>
#include <iostream>
#include <cassert>
#include <random>
#include <numeric>

namespace pluto {
    auto tensor::create(context* const ctx, const std::span<const dim> dims) noexcept -> pool_ref<tensor> {
        assert(ctx != nullptr);
        pool_ref<tensor> t {ctx->pool_alloc<tensor>()}; // Allocate memory for the tensor
        t->m_ctx = ctx;
        constexpr dim scalar_size {sizeof(float)};
        dim size {static_cast<dim>(std::accumulate(dims.begin(), dims.end(), scalar_size, std::multiplies<>{}))};
        auto* const buf {static_cast<float*>(ctx->pool_alloc_raw_aligned(size, buf_align))};
        t->m_buf = {buf, static_cast<std::size_t>(size / scalar_size)}; // Allocate memory for the data
        std::fill(t->m_buf.begin(), t->m_buf.end(), 0); // Zero-initialize the data TODO: remove
        t->m_shape = tensor_shape<float> {dims};
        return t;
    }

    auto tensor::create(context* ctx, const std::initializer_list<const dim> dims) noexcept -> pool_ref<tensor> {
        return create(ctx, std::span<const dim>{dims});
    }

    auto tensor::isomorphic_clone() const -> pool_ref<tensor> {
        return create(m_ctx, static_cast<std::span<const dim>>(m_shape));
    }

    auto tensor::deep_clone() const -> pool_ref<tensor> {
        pool_ref<tensor> t {this->isomorphic_clone()};
        std::copy(m_buf.begin(), m_buf.end(), t->m_buf.begin());
        return t;
    }

    auto tensor::fill(const float val) noexcept -> void {
        std::fill(m_buf.begin(), m_buf.end(), val);
    }

    auto tensor::populate(const std::span<const float> values) noexcept -> void {
        assert(m_buf.size() == values.size());
        std::copy(values.begin(), values.end(), m_buf.begin());
    }

    auto tensor::get_args() const noexcept -> std::span<const pool_ref<tensor>> { return {m_args.data(), m_num_args}; }

    auto tensor::get_args() noexcept -> std::span<pool_ref<tensor>> { return {m_args.data(), m_num_args}; }

    auto tensor::get_op_code() const noexcept -> opcode { return m_op; }

    auto tensor::is_leaf_node() const noexcept -> bool { return m_op == opcode::nop; }

    auto tensor::push_arg(const pool_ref<tensor> t) -> void {
        assert(m_num_args < max_args);
        m_args[m_num_args++] = t;
    }

    static thread_local std::random_device rnd_dvc {};
    static thread_local std::mt19937_64 rnd_gen {};

    auto tensor::fill_random(const float min, const float max) noexcept -> void {
        std::uniform_real_distribution<float> dist {min, max};
        fill_fn([&dist](dim) noexcept -> float  {
            return dist(rnd_gen);
        });
    }

    auto operator << (std::ostream& o, const tensor& self) -> std::ostream& {
        o << "[\n";
        for (dim i3 {}; i3 < self.m_shape[2]; ++i3) {
            for (dim i2 {}; i2 < self.m_shape[1]; ++i2) {
                o << '\t';
                for (dim i1 {}; i1 < self.m_shape[0]; ++i1) {
                    o << self.m_buf[i3*self.m_shape[1]*self.m_shape[0] + i2*self.m_shape[0] + i1] << ' ';
                }
                o << '\n';
            }
        }
        o << "]\n";
        return o;
    }
}
