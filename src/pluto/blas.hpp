// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <cstdint>

#ifdef __ARM_NEON
#   include <arm_neon.h>
#endif
#if defined(_MSC_VER) || defined(__MINGW32__)
#   include <intrin.h>
#elif defined(__x86_64__) || defined(_M_AMD64)
#   include <immintrin.h>
#endif

namespace pluto {
    // IEEE 754 754-2008 binary 16 (half precision float)
    struct f16 final {
        std::uint16_t bits {};

        constexpr f16() noexcept = default;
        inline explicit f16(const float x) noexcept {
            #if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
                const __fp16 f16 = static_cast<__fp16>(x);
                bits = *reinterpret_cast<const std::uint16_t*>(&f16);
            #else
                float base = (std::abs(x) * 0x1.0p+112f) * 0x1.0p-110f;  // Normalize |x|
                const std::uint32_t w = *reinterpret_cast<const std::uint32_t*>(&x);
                const std::uint32_t shl1_w = w + w;
                const std::uint32_t sign = w & 0x80000000u;
                std::uint32_t bias = shl1_w & 0xff000000u; // Extract bias
                if (bias < 0x71000000u) bias = 0x71000000u; // Apply minimum bias for subnormals
                bias = (bias>>1) + 0x07800000u; // Adjust bias for half precision
                base = *reinterpret_cast<float*>(&bias) + base;
                const std::uint32_t rbits = *reinterpret_cast<const std::uint32_t*>(&base); // Extract bits
                const std::uint32_t exp_bits = (rbits>>13) & 0x00007c00u; // Extract exponent bits
                const std::uint32_t mant_bits = rbits & 0x00000fffu; // Extract mantissa bits
                const std::uint32_t nonsign = exp_bits + mant_bits; // Combine exponent and mantissa bits
                bits = (sign>>16) | (shl1_w > 0xff000000 ? 0x7e00 : nonsign); // Pack full bit pattern
            #endif
        }

        inline explicit operator float() const noexcept {
            #if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
                return static_cast<float>(*reinterpret_cast<const __fp16*>(&bits));
            #else // Slow software emulated path
                const std::uint32_t w = static_cast<std::uint32_t>(bits)<<16;
                const std::uint32_t sign = w & 0x80000000u;
                const std::uint32_t two_w = w + w;
                const std::uint32_t exp_offset = 0xe0u<<23; // Exponent offset for normalization
                std::uint32_t tmp = (two_w>>4) + exp_offset; // Adjust exponent
                const float norm_x = *reinterpret_cast<float*>(&tmp) * 0x1.0p-112f; // Normalize the result
                tmp = (two_w>>17) | (126u<<23); // Adjust exponent for denormalized values
                const float denorm_x = *reinterpret_cast<float*>(&tmp) - 0.5f;
                const std::uint32_t denorm_cutoff = 1u<<27; // Threshold for denormalized values
                const std::uint32_t result = sign // Combine sign and mantissa
                    | (two_w < denorm_cutoff
                    ? *reinterpret_cast<const std::uint32_t*>(&denorm_x) // Use denormalized value if below cutoff
                    : *reinterpret_cast<const std::uint32_t*>(&norm_x)); // Else use normalized value
                return *(float *)&result;
            #endif
        }

        [[nodiscard]] static inline auto e() noexcept -> f16 { return f16{0x4170}; }
        [[nodiscard]] static inline auto eps() noexcept -> f16 { return f16{0x1400}; }
        [[nodiscard]] static inline auto inf() noexcept -> f16 { return f16{0x7c00}; }
        [[nodiscard]] static inline auto ln_10() noexcept -> f16 { return f16{0x409b}; }
        [[nodiscard]] static inline auto ln_2() noexcept -> f16 { return f16{0x398c}; }
        [[nodiscard]] static inline auto log10_2() noexcept -> f16 { return f16{0x34d1}; }
        [[nodiscard]] static inline auto log10_e() noexcept -> f16 { return f16{0x36f3}; }
        [[nodiscard]] static inline auto log2_10() noexcept -> f16 { return f16{0x42a5}; }
        [[nodiscard]] static inline auto log2_e() noexcept -> f16 { return f16{0x3dc5}; }
        [[nodiscard]] static inline auto max() noexcept -> f16 { return f16{0x7bff}; }
        [[nodiscard]] static inline auto max_subnormal() noexcept -> f16 { return f16{0x03ff}; }
        [[nodiscard]] static inline auto min() noexcept -> f16 { return f16{0xfbff}; }
        [[nodiscard]] static inline auto min_pos() noexcept -> f16 { return f16{0x0400}; }
        [[nodiscard]] static inline auto min_pos_subnormal() noexcept -> f16 { return f16{0x0001}; }
        [[nodiscard]] static inline auto nan() noexcept -> f16 { return f16{0x7e00}; }
        [[nodiscard]] static inline auto neg_inf() noexcept -> f16 { return f16{0xfc00}; }
        [[nodiscard]] static inline auto neg_one() noexcept -> f16 { return f16{0xbc00}; }
        [[nodiscard]] static inline auto neg_zero() noexcept -> f16 { return f16{0x8000}; }
        [[nodiscard]] static inline auto one() noexcept -> f16 { return f16{0x3c00}; }
        [[nodiscard]] static inline auto pi() noexcept -> f16 { return f16{0x4248}; }
        [[nodiscard]] static inline auto sqrt_2() noexcept -> f16 { return f16{0x3da8}; }
        [[nodiscard]] static inline auto zero() noexcept -> f16 { return f16{0x0000}; }
    };
    static_assert(sizeof(f16) == 2);

    // Google brain float 16 (bfloat16) format
    struct bf16 final {
        std::uint16_t bits {};

        constexpr bf16() noexcept = default;
        inline explicit bf16(const float x) noexcept {
            if (((*reinterpret_cast<const std::uint32_t*>(&x)) & 0x7fffffff) > 0x7f800000) { // NaN
                bits = 64 | ((*(uint32_t *)&x)>>16); // quiet NaNs only
            }
            if (!((*reinterpret_cast<const std::uint32_t*>(&x)) & 0x7f800000)) { // Subnormals
                bits = ((*reinterpret_cast<const std::uint32_t*>(&x)) & 0x80000000)>>16; // Flush to zero
            }
            bits = ((*reinterpret_cast<const std::uint32_t*>(&x))+ (0x7fff
                + (((*reinterpret_cast<const std::uint32_t*>(&x))>>16) & 1)))>>16; // Rounding and composing final bf16 value
        }

        inline explicit operator float() const noexcept {
            const auto tmp = static_cast<std::uint32_t>(bits)<<16; // bf16 is basically a truncated f32
            return *reinterpret_cast<const float*>(&tmp);
        }

        [[nodiscard]] static inline auto eps() noexcept -> bf16 { return bf16{0x3c00}; }
        [[nodiscard]] static inline auto inf() noexcept -> bf16 { return bf16{0x7f80}; }
        [[nodiscard]] static inline auto max() noexcept -> bf16 { return bf16{0x7f7f}; }
        [[nodiscard]] static inline auto min() noexcept -> bf16 { return bf16{0xff7f}; }
        [[nodiscard]] static inline auto min_pos() noexcept -> bf16 { return bf16{0x0080}; }
        [[nodiscard]] static inline auto nan() noexcept -> bf16 { return bf16{0x7FC0}; }
        [[nodiscard]] static inline auto neg_inf() noexcept -> bf16 { return bf16{0xff80}; }
        [[nodiscard]] static inline auto min_pos_subnormal() noexcept -> bf16 { return bf16{0x0001}; }
        [[nodiscard]] static inline auto max_subnormal() noexcept -> bf16 { return bf16{0x007f}; }
        [[nodiscard]] static inline auto one() noexcept -> bf16 { return bf16{0x3f80}; }
        [[nodiscard]] static inline auto zero() noexcept -> bf16 { return bf16{0x0000}; }
        [[nodiscard]] static inline auto neg_zero() noexcept -> bf16 { return bf16{0x8000}; }
        [[nodiscard]] static inline auto neg_one() noexcept -> bf16 { return bf16{0xbf80}; }
        [[nodiscard]] static inline auto e() noexcept -> bf16 { return bf16{0x402e}; }
        [[nodiscard]] static inline auto pi() noexcept -> bf16 { return bf16{0x4049}; }
        [[nodiscard]] static inline auto ln_10() noexcept -> bf16 { return bf16{0x4013}; }
        [[nodiscard]] static inline auto ln_2() noexcept -> bf16 { return bf16{0x3f31}; }
        [[nodiscard]] static inline auto log10_e() noexcept -> bf16 { return bf16{0x3ede}; }
        [[nodiscard]] static inline auto log10_2() noexcept -> bf16 { return bf16{0x3e9a}; }
        [[nodiscard]] static inline auto log2_e() noexcept -> bf16 { return bf16{0x3fb9}; }
        [[nodiscard]] static inline auto log2_10() noexcept -> bf16 { return bf16{0x4055}; }
        [[nodiscard]] static inline auto sqrt_2() noexcept -> bf16 { return bf16{0x3fb5}; }
    };
    static_assert(sizeof(bf16) == 2);
}
