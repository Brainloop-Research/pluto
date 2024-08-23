// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "f16.hpp"

#include <bit>

#ifdef __ARM_NEON
#   include <arm_neon.h>
#endif
#if defined(_MSC_VER) || defined(__MINGW32__)
#   include <intrin.h>
#elif defined(__x86_64__) || defined(_M_AMD64)
#   include <immintrin.h>
#endif

namespace pluto {
    f16::f16(const float x) noexcept {
        #if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
            const __fp16 f16 {static_cast<__fp16>(x)};
            bits = std::bit_cast<std::uint16_t>(f16);
        #elif defined(__F16C__) // Fast hardware path
            #ifdef _MSC_VER
                bits = static_cast<std::uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0));
            #else
                bits = static_cast<std::uint16_t>(_cvtss_sh(x, 0));
            #endif
        #else // Slow software emulated path
            const float base {(std::abs(x) * 0x1.0p+112f) * 0x1.0p-110f};  // Normalize |x|
            const std::uint32_t w {std::bit_cast<std::uint32_t>(x)};
            const std::uint32_t shl1_w {w+w};
            const std::uint32_t sign {w & 0x80000000u};
            const std::uint32_t bias {0x07800000u+(std::max(0x71000000u, shl1_w&0xff000000u)>>1)}; // Extract bias
            const std::uint32_t rbits {std::bit_cast<std::uint32_t>(base + std::bit_cast<float>(bias))}; // Extract bits
            const std::uint32_t exp_bits {(rbits>>13) & 0x00007c00u}; // Extract exponent bits
            const std::uint32_t mant_bits {rbits & 0x00000fffu}; // Extract mantissa bits
            const std::uint32_t nonsign {exp_bits + mant_bits}; // Combine exponent and mantissa bits
            bits = (sign>>16)|(shl1_w > 0xff000000 ? 0x7e00 : nonsign); // Pack full bit pattern
        #endif
    }

    f16::operator float() const noexcept {
        #if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
            return static_cast<float>(std::bit_cast<__fp16>(bits));
        #elif defined(__F16C__) // Fast hardware path
            #ifdef _MSC_VER
                return static_cast<float>(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits))));
            #else
                return static_cast<float>(_cvtsh_ss(bits));
            #endif
        #else // Slow software emulated path
            const std::uint32_t w {static_cast<std::uint32_t>(bits)<<16};
            const std::uint32_t sign {w & 0x80000000u};
            const std::uint32_t two_w {w+w};
            const std::uint32_t exp_offset {0xe0u<<23}; // Exponent offset for normalization
            const float norm_x {std::bit_cast<float>((two_w>>4) + exp_offset) * 0x1.0p-112f}; // Normalize the result
            const float denorm_x {std::bit_cast<float>((two_w>>17) | (126u<<23)) - 0.5f}; // Adjust exponent for denormalized values
            const std::uint32_t denorm_cutoff {1u<<27}; // Threshold for denormalized values
            const std::uint32_t result = sign // Combine sign and mantissa
                | (two_w < denorm_cutoff
                ? std::bit_cast<std::uint32_t>(denorm_x) // Use denormalized value if below cutoff
                : std::bit_cast<std::uint32_t>(norm_x)); // Else use normalized value
            return std::bit_cast<float>(result);
        #endif
    }
}
