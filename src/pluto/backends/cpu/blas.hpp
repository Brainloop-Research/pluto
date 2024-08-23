// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "../../tensor.hpp"

namespace pluto {
    struct f16;
    struct bf16;
}

namespace pluto::backends::cpu::blas {

    // ---- Vector Operations ----

    extern auto v_cvt_f16_to_f32(dim n, float* o, const f16* x) noexcept -> void;
    extern auto v_cvt_f32_to_f16(dim n, f16* o, const float* x) noexcept -> void;
    extern auto v_cvt_bf16_to_f32(dim n, float* o, const bf16* x) noexcept -> void;
    extern auto v_cvt_f32_to_bf16(dim n, bf16* o, const float* x) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_softmax(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_softmax(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_sigmoid(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_sigmoid(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_tanh(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_tanh(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_relu(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_relu(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_gelu(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_gelu(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_silu(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_silu(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_add(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x,
        const T* __restrict__ y
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_add(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x,
        const float* __restrict__ y
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_sub(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x,
        const T* __restrict__ y
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_sub(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x,
        const float* __restrict__ y
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_mul(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x,
        const T* __restrict__ y
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_mul(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x,
        const float* __restrict__ y
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    extern auto PT_HOTPROC v_div(
        dim n,
        T* __restrict__ o,
        const T* __restrict__ x,
        const T* __restrict__ y
    ) noexcept -> void;
    template <>
    auto PT_HOTPROC v_div(
        dim n,
        float* __restrict__ o,
        const float* __restrict__ x,
        const float* __restrict__ y
    ) noexcept -> void;

    template <typename T> requires is_dtype<T>
    [[nodiscard]] extern auto PT_HOTPROC v_dot(
        dim n,
        const T* __restrict__ x,
        const T* __restrict__ y
    ) noexcept -> T;
    template <>
    [[nodiscard]] auto PT_HOTPROC v_dot(
        dim n,
        const float* __restrict__ x,
        const float* __restrict__ y
    ) noexcept -> float;

    // ---- Tensor Operations ----

    [[nodiscard]] extern auto t_softmax(const compute_ctx& ctx, const tensor& x) noexcept -> tensor*;
    [[nodiscard]] extern auto t_sigmoid(const compute_ctx& ctx, const tensor& x) noexcept -> tensor*;
    [[nodiscard]] extern auto t_tanh(const compute_ctx& ctx, const tensor& x) noexcept -> tensor*;
    [[nodiscard]] extern auto t_relu(const compute_ctx& ctx, const tensor& x) noexcept -> tensor*;
    [[nodiscard]] extern auto t_gelu(const compute_ctx& ctx, const tensor& x) noexcept -> tensor*;
    [[nodiscard]] extern auto t_silu(const compute_ctx& ctx, const tensor& x) noexcept -> tensor*;

    [[nodiscard]] extern auto t_add(const compute_ctx& ctx, const tensor& x, const tensor& y) noexcept -> tensor*;
    [[nodiscard]] extern auto t_sub(const compute_ctx& ctx, const tensor& x, const tensor& y) noexcept -> tensor*;
    [[nodiscard]] extern auto t_mul(const compute_ctx& ctx, const tensor& x, const tensor& y) noexcept -> tensor*;
    [[nodiscard]] extern auto t_div(const compute_ctx& ctx, const tensor& x, const tensor& y) noexcept -> tensor*;
    [[nodiscard]] extern auto t_dot(const compute_ctx& ctx, const tensor& x, const tensor& y) noexcept -> tensor*;
}