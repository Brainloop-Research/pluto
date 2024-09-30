// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "backends/cpu/blas.hpp"

namespace pluto {

    // IEEE 754 754-2008 binary 16 (half precision float)
    struct f16 final {
        std::uint16_t bits {};

        constexpr f16() noexcept = default;
        constexpr explicit f16(const int x) noexcept : bits{static_cast<std::uint16_t>(x)} {}
        inline explicit f16(const float x) noexcept {
            backends::cpu::blas::v_cvt_f32_to_f16(1, this, &x);
        }
        inline explicit operator float() const noexcept {
            float r;
            backends::cpu::blas::v_cvt_f16_to_f32(1, &r, this);
            return r;
        }
        [[nodiscard]] static constexpr auto e() noexcept -> f16 { return f16{0x4170}; }
        [[nodiscard]] static constexpr auto eps() noexcept -> f16 { return f16{0x1400}; }
        [[nodiscard]] static constexpr auto inf() noexcept -> f16 { return f16{0x7c00}; }
        [[nodiscard]] static constexpr auto ln_10() noexcept -> f16 { return f16{0x409b}; }
        [[nodiscard]] static constexpr auto ln_2() noexcept -> f16 { return f16{0x398c}; }
        [[nodiscard]] static constexpr auto log10_2() noexcept -> f16 { return f16{0x34d1}; }
        [[nodiscard]] static constexpr auto log10_e() noexcept -> f16 { return f16{0x36f3}; }
        [[nodiscard]] static constexpr auto log2_10() noexcept -> f16 { return f16{0x42a5}; }
        [[nodiscard]] static constexpr auto log2_e() noexcept -> f16 { return f16{0x3dc5}; }
        [[nodiscard]] static constexpr auto max() noexcept -> f16 { return f16{0x7bff}; }
        [[nodiscard]] static constexpr auto max_subnormal() noexcept -> f16 { return f16{0x03ff}; }
        [[nodiscard]] static constexpr auto min() noexcept -> f16 { return f16{0xfbff}; }
        [[nodiscard]] static constexpr auto min_pos() noexcept -> f16 { return f16{0x0400}; }
        [[nodiscard]] static constexpr auto min_pos_subnormal() noexcept -> f16 { return f16{0x0001}; }
        [[nodiscard]] static constexpr auto nan() noexcept -> f16 { return f16{0x7e00}; }
        [[nodiscard]] static constexpr auto neg_inf() noexcept -> f16 { return f16{0xfc00}; }
        [[nodiscard]] static constexpr auto neg_one() noexcept -> f16 { return f16{0xbc00}; }
        [[nodiscard]] static constexpr auto neg_zero() noexcept -> f16 { return f16{0x8000}; }
        [[nodiscard]] static constexpr auto one() noexcept -> f16 { return f16{0x3c00}; }
        [[nodiscard]] static constexpr auto pi() noexcept -> f16 { return f16{0x4248}; }
        [[nodiscard]] static constexpr auto sqrt_2() noexcept -> f16 { return f16{0x3da8}; }
        [[nodiscard]] static constexpr auto zero() noexcept -> f16 { return f16{0x0000}; }

        inline auto operator ==(const f16 rhs) const noexcept -> bool { // Epsilon comparison: |ξ1 - ξ2| < ε
            const auto xi1 {static_cast<float>(* this)};
            const auto xi2 {static_cast<float>(rhs)};
            const auto epsi {static_cast<float>(eps())};
            return std::abs(xi1 - xi2) < epsi;
        }
        inline auto operator !=(const f16 rhs) const noexcept -> bool {
            return !(*this == rhs);
        }
        inline auto operator ==(const float xi2) const noexcept -> bool { // Epsilon comparison: |ξ1 - ξ2| < ε
            const auto xi1 {static_cast<float>(* this)};
            const auto epsi {static_cast<float>(eps())};
            return std::abs(xi1 - xi2) < epsi;
        }
        inline auto operator !=(const float rhs) const noexcept -> bool {
            return !(*this == rhs);
        }
    };
    static_assert(sizeof(f16) == 2);
}
