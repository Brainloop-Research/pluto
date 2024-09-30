// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "backends/cpu/blas.hpp"

namespace pluto {
    // Google brain float 16 (bfloat16) format
    struct bf16 final {
        std::uint16_t bits {};

        constexpr bf16() noexcept = default;
        constexpr explicit bf16(const int x) noexcept : bits{static_cast<std::uint16_t>(x)} {}
        inline explicit bf16(const float x) noexcept {
            backends::cpu::blas::v_cvt_f32_to_bf16(1, this, &x);
        }

        inline explicit operator float() const noexcept {
            float r;
            backends::cpu::blas::v_cvt_bf16_to_f32(1, &r, this);
            return r;
        }

        [[nodiscard]] static constexpr auto eps() noexcept -> bf16 { return bf16{0x3c00}; }
        [[nodiscard]] static constexpr auto inf() noexcept -> bf16 { return bf16{0x7f80}; }
        [[nodiscard]] static constexpr auto max() noexcept -> bf16 { return bf16{0x7f7f}; }
        [[nodiscard]] static constexpr auto min() noexcept -> bf16 { return bf16{0xff7f}; }
        [[nodiscard]] static constexpr auto min_pos() noexcept -> bf16 { return bf16{0x0080}; }
        [[nodiscard]] static constexpr auto nan() noexcept -> bf16 { return bf16{0x7FC0}; }
        [[nodiscard]] static constexpr auto neg_inf() noexcept -> bf16 { return bf16{0xff80}; }
        [[nodiscard]] static constexpr auto min_pos_subnormal() noexcept -> bf16 { return bf16{0x0001}; }
        [[nodiscard]] static constexpr auto max_subnormal() noexcept -> bf16 { return bf16{0x007f}; }
        [[nodiscard]] static constexpr auto one() noexcept -> bf16 { return bf16{0x3f80}; }
        [[nodiscard]] static constexpr auto zero() noexcept -> bf16 { return bf16{0x0000}; }
        [[nodiscard]] static constexpr auto neg_zero() noexcept -> bf16 { return bf16{0x8000}; }
        [[nodiscard]] static constexpr auto neg_one() noexcept -> bf16 { return bf16{0xbf80}; }
        [[nodiscard]] static constexpr auto e() noexcept -> bf16 { return bf16{0x402e}; }
        [[nodiscard]] static constexpr auto pi() noexcept -> bf16 { return bf16{0x4049}; }
        [[nodiscard]] static constexpr auto ln_10() noexcept -> bf16 { return bf16{0x4013}; }
        [[nodiscard]] static constexpr auto ln_2() noexcept -> bf16 { return bf16{0x3f31}; }
        [[nodiscard]] static constexpr auto log10_e() noexcept -> bf16 { return bf16{0x3ede}; }
        [[nodiscard]] static constexpr auto log10_2() noexcept -> bf16 { return bf16{0x3e9a}; }
        [[nodiscard]] static constexpr auto log2_e() noexcept -> bf16 { return bf16{0x3fb9}; }
        [[nodiscard]] static constexpr auto log2_10() noexcept -> bf16 { return bf16{0x4055}; }
        [[nodiscard]] static constexpr auto sqrt_2() noexcept -> bf16 { return bf16{0x3fb5}; }

        inline auto operator ==(const bf16 rhs) const noexcept -> bool { // Epsilon comparison: |ξ1 - ξ2| < ε
            const auto xi1 {static_cast<float>(*this)};
            const auto xi2 {static_cast<float>(rhs)};
            const auto epsi {static_cast<float>(eps())};
            return std::abs(xi1 - xi2) < epsi;
        }
        inline auto operator !=(const bf16 rhs) const noexcept -> bool {
            return !(*this == rhs);
        }
        inline auto operator ==(const float xi2) const noexcept -> bool { // Epsilon comparison: |ξ1 - ξ2| < ε
            const auto xi1 {static_cast<float>(*this)};
            const auto epsi {static_cast<float>(eps())};
            return std::abs(xi1 - xi2) < epsi;
        }
        inline auto operator !=(const float rhs) const noexcept -> bool {
            return !(*this == rhs);
        }
    };
    static_assert(sizeof(bf16) == 2);
}
