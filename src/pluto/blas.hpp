// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <bit>
#include <array>
#include <algorithm>
#include <numbers>
#include <cstdint>
#include <cmath>

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
        constexpr explicit f16(const int x) noexcept : bits{static_cast<std::uint16_t>(x)} {}
        inline explicit f16(const float x) noexcept {
            #if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
                const auto f16 = static_cast<__fp16>(x);
                bits = std::bit_cast<std::uint16_t>(f16);
            #elif defined(__F16C__) // Fast hardware path
            #   ifdef _MSC_VER
                    bits = static_cast<std::uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0));
            #   else
                    bits = static_cast<std::uint16_t>(_cvtss_sh(x, 0));
            #   endif
            #else // Slow software emulated path
                const float base = (std::abs(x) * 0x1.0p+112f) * 0x1.0p-110f;  // Normalize |x|
                const std::uint32_t w = std::bit_cast<std::uint32_t>(x);
                const std::uint32_t shl1_w = w + w;
                const std::uint32_t sign = w & 0x80000000u;
                const std::uint32_t bias = 0x07800000u+(std::max(0x71000000u, shl1_w&0xff000000u)>>1); // Extract bias
                const std::uint32_t rbits = std::bit_cast<std::uint32_t>(base + std::bit_cast<float>(bias)); // Extract bits
                const std::uint32_t exp_bits = (rbits>>13) & 0x00007c00u; // Extract exponent bits
                const std::uint32_t mant_bits = rbits & 0x00000fffu; // Extract mantissa bits
                const std::uint32_t nonsign = exp_bits + mant_bits; // Combine exponent and mantissa bits
                bits = (sign>>16)|(shl1_w > 0xff000000 ? 0x7e00 : nonsign); // Pack full bit pattern
            #endif
        }

        inline explicit operator float() const noexcept {
            #if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
                return static_cast<float>(std::bit_cast<__fp16>(bits));
            #elif defined(__F16C__) // Fast hardware path
            #   ifdef _MSC_VER
                    return static_cast<float>(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits))));
            #   else
                    return static_cast<float>(_cvtsh_ss(bits));
            #   endif
            #else // Slow software emulated path
                const std::uint32_t w = static_cast<std::uint32_t>(bits)<<16;
                const std::uint32_t sign = w & 0x80000000u;
                const std::uint32_t two_w = w + w;
                const std::uint32_t exp_offset = 0xe0u<<23; // Exponent offset for normalization
                const float norm_x = std::bit_cast<float>((two_w>>4) + exp_offset) * 0x1.0p-112f; // Normalize the result
                const float denorm_x = std::bit_cast<float>((two_w>>17) | (126u<<23)) - 0.5f; // Adjust exponent for denormalized values
                const std::uint32_t denorm_cutoff = 1u<<27; // Threshold for denormalized values
                const std::uint32_t result = sign // Combine sign and mantissa
                    | (two_w < denorm_cutoff
                    ? std::bit_cast<std::uint32_t>(denorm_x) // Use denormalized value if below cutoff
                    : std::bit_cast<std::uint32_t>(norm_x)); // Else use normalized value
                return std::bit_cast<float>(result);
            #endif
        }

        static inline auto cvt_f16_to_f32_vec(const std::int64_t n, float* const o, const f16* const x) noexcept -> void {
            std::int64_t i {};
            #ifdef __ARM_NEON
                for (; i+7 < n; i += 8) {
                    const float16x8_t v0 {vld1q_f16(reinterpret_cast<const float16_t*>(x+i))};
                    const float32x4_t f0 {vcvt_f32_f16(vget_low_f16(v0))};
                    const float32x4_t f1 {vcvt_f32_f16(vget_high_f16(v0))};
                    vst1q_f32(o+i, f0);
                    vst1q_f32(o+i+4, f1);
                }
                for (; i+3 < n; i += 4) {
                    float16x4_t v {vld1_f16(reinterpret_cast<const float16_t*>(x+i))};
                    float32x4_t f {vcvt_f32_f16(v)};
                    vst1q_f32(o+i, f);
                }
            #endif
            for (; i < n; ++i) {
                o[i] = static_cast<float>(x[i]);
            }
        }

        static inline auto cvt_f32_to_f16_vec(const std::int64_t n, f16* const o, const float* const x) noexcept -> void {
            std::int64_t i {};
            #ifdef __F16C__
                for (; i+7 < n; i += 8) {
                    _mm_storeu_si128(
                        reinterpret_cast<__m128i*>(o+i),
                        _mm256_cvtps_ph(
                            _mm256_loadu_ps(x+i),
                            _MM_FROUND_TO_NEAREST_INT
                        )
                    );
                }
                for(; i+3 < n; i += 4) {
                    _mm_storel_epi64(
                        reinterpret_cast<__m128i*>(o+i),
                        _mm_cvtps_ph(
                            _mm_loadu_ps(x+i),
                            _MM_FROUND_TO_NEAREST_INT
                        )
                    );
                }
            #elif defined (__ARM_NEON)
                for (; i+7 < n; i += 8) {
                    const float32x4_t v0 {vld1q_f32(x+i)};
                    const float32x4_t v1 {vld1q_f32(x+i+4)};
                    const float16x4_t h0 {vcvt_f16_f32(v0)};
                    const float16x4_t h1 {vcvt_f16_f32(v1)};
                    vst1_f16(reinterpret_cast<float16_t*>(o+i), h0);
                    vst1_f16(reinterpret_cast<float16_t*>(o+i+4), h1);
                }
                for (; i+3 < n; i += 4) {
                    const float32x4_t v {vld1q_f32(x+i)};
                    const float16x4_t h {vcvt_f16_f32(v)};
                    vst1_f16(reinterpret_cast<float16_t*>(o+i), h);
                }
            #endif
            for (; i < n; ++i) {
                o[i] = f16{x[i]};
            }
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
            const auto xi1 = static_cast<float>(*this);
            const auto xi2 = static_cast<float>(rhs);
            const auto epsi = static_cast<float>(eps());
            return std::abs(xi1 - xi2) < epsi;
        }
        inline auto operator !=(const f16 rhs) const noexcept -> bool {
            return !(*this == rhs);
        }
        inline auto operator ==(const float xi2) const noexcept -> bool { // Epsilon comparison: |ξ1 - ξ2| < ε
            const auto xi1 = static_cast<float>(*this);
            const auto epsi = static_cast<float>(eps());
            return std::abs(xi1 - xi2) < epsi;
        }
        inline auto operator !=(const float rhs) const noexcept -> bool {
            return !(*this == rhs);
        }
    };
    static_assert(sizeof(f16) == 2);

    // Google brain float 16 (bfloat16) format
    struct bf16 final {
        std::uint16_t bits {};

        constexpr bf16() noexcept = default;
        constexpr explicit bf16(const int x) noexcept : bits{static_cast<std::uint16_t>(x)} {}
        constexpr explicit bf16(const float x) noexcept {
            const auto bi {std::bit_cast<std::uint32_t>(x)};
            if ((bi & 0x7fffffff) > 0x7f800000) { // NaN
                bits = 64 | (bi>>16); // quiet NaNs only
            }
            if (!(bi & 0x7f800000)) { // Subnormals
                bits = (bi & 0x80000000)>>16; // Flush to zero
            }
            bits = (bi + (0x7fff + ((bi>>16) & 1)))>>16; // Rounding and composing final bf16 value
        }

        constexpr explicit operator float() const noexcept {
            const auto tmp = static_cast<std::uint32_t>(bits)<<16; // bf16 is basically a truncated f32
            return std::bit_cast<float>(tmp);
        }

        static inline auto cvt_bf16_to_f32_vec(const std::int64_t n, float* const o, const bf16* const x) noexcept -> void {
            std::int64_t i {};
            #ifdef __AVX512F__
                for (; i+15 < n; i += 16) {
                    _mm512_storeu_ps(o+i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+i))
                                ), 16
                            )
                        )
                    );
                }
            #elif defined(__AVX2__)
                for (; i+7 < n; i += 8) {
                    _mm256_storeu_ps(
                        o+i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i))
                                ), 16
                            )
                        )
                    );
                }
            #elif defined(__ARM_NEON)
                for (; i+7 < n; i += 8) {
                    const uint16x8_t vbf16 {vld1q_u16(reinterpret_cast<const std::uint16_t*>(x+i))};
                    const uint32x4_t t_lo {vshlq_n_u32(vmovl_u16(vget_low_u16(vbf16)), 16)};
                    const uint32x4_t t_hi {vshlq_n_u32(vmovl_u16(vget_high_u16(vbf16)), 16)};
                    const float32x4_t f32_lo {vreinterpretq_f32_u32(t_lo)};
                    const float32x4_t f32_hi {vreinterpretq_f32_u32(t_hi)};
                    vst1q_f32(o+i, f32_lo);
                    vst1q_f32(o+i+4, f32_hi);
                }
                for (; i+3 < n; i += 4) {
                    const uint16x4_t vbf16 {vld1_u16(reinterpret_cast<const std::uint16_t*>(x+i))};
                    const uint32x4_t tx {vshlq_n_u32(vmovl_u16(vbf16), 16)};
                    const float32x4_t f32 {vreinterpretq_f32_u32(tx)};
                    vst1q_f32(o+i, f32);
                }
            #endif
            for (; i < n; ++i) {
                o[i] = static_cast<float>(x[i]);
            }
        }

        static inline auto cvt_f32_to_bf16_vec(const std::int64_t n, bf16* const o, const float* const x) noexcept -> void {
            std::int64_t i {};
            #ifdef __AVX512BF16__
                for (; i+31 < n; i += 32) {
                    _mm512_storeu_si512(
                        reinterpret_cast<__m512i*>(y+i),
                        __m512i{
                            _mm512_cvtne2ps_pbh(
                                _mm512_loadu_ps(x+i+16),
                                _mm512_loadu_ps(x+i)
                            )
                        }
                    );
                }
            #endif
            for (; i < n; ++i) {
                o[i] = bf16{x[i]};
            }
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
            const auto xi1 = static_cast<float>(*this);
            const auto xi2 = static_cast<float>(rhs);
            const auto epsi = static_cast<float>(eps());
            return std::abs(xi1 - xi2) < epsi;
        }
        inline auto operator !=(const bf16 rhs) const noexcept -> bool {
            return !(*this == rhs);
        }
        inline auto operator ==(const float xi2) const noexcept -> bool { // Epsilon comparison: |ξ1 - ξ2| < ε
            const auto xi1 = static_cast<float>(*this);
            const auto epsi = static_cast<float>(eps());
            return std::abs(xi1 - xi2) < epsi;
        }
        inline auto operator !=(const float rhs) const noexcept -> bool {
            return !(*this == rhs);
        }
    };
    static_assert(sizeof(bf16) == 2);

    namespace vblas { // Vector BLAS
        #define PT_X86_X64_USE_HADD

        template <typename T>
        inline auto dot(std::int64_t n, const T* x, const T* y) noexcept -> T;

        template <>
        inline auto dot(const std::int64_t n, const float* __restrict__ const x, const float* __restrict__ const y) noexcept -> float {
        #ifdef __AVX512F__
            constexpr std::int64_t step {64};
            const std::int64_t k {n & -step};
            __m512 acc[4] = {_mm512_setzero_ps()};
            __m512 vx[4];
            __m512 vy[4];
            for (std::int64_t i {}; i < k; i += step) {
                #pragma GCC unroll 4
                for (std::int64_t j {}; j < 4; ++j) {
                    vx[j] = _mm512_loadu_ps(x+i+(j<<4));
                    vy[j] = _mm512_loadu_ps(y+i+(j<<4));
                    acc[j] = _mm512_fmadd_ps(vx[j], vy[j], acc[j]);
                }
            }
            acc[1] = _mm512_add_ps(acc[1], acc[3]);
            *acc = _mm512_add_ps(*acc, acc[2]);
            *acc = _mm512_add_ps(*acc, acc[1]);
            return _mm512_reduce_add_ps(*acc);
        #elif defined(__AVX__) && defined(__FMA__)
            constexpr std::int64_t step {32};
            const std::int64_t k {n & -step};
            __m256 acc[4] {_mm256_setzero_ps()};
            __m256 vx[4];
            __m256 vy[4];
            for (std::int64_t i {}; i < k; i += step) {
                #pragma GCC unroll 4
                for (std::int64_t j {}; j < 4; ++j) {
                    vx[j] = _mm256_loadu_ps(x+i+(j<<3));
                    vy[j] = _mm256_loadu_ps(y+i+(j<<3));
                    acc[j] = _mm256_fmadd_ps(vx[j], vy[j], acc[j]);
                }
            }
            acc[1] = _mm256_add_ps(acc[1], acc[3]);
            *acc = _mm256_add_ps(*acc, acc[2]);
            *acc = _mm256_add_ps(*acc, acc[1]);
            float sum;
            #ifdef PT_X86_X64_USE_HADD
                __m128 v0 {_mm_add_ps(_mm256_castps256_ps128(*acc), _mm256_extractf128_ps(*acc, 1))};
                v0 = _mm_hadd_ps(v0, v0);
                v0 = _mm_hadd_ps(v0, v0);
                sum = _mm_cvtss_f32(v0);
            #else
                const __m128 xmm0 {_mm256_castps256_ps128(*acc)};
                const __m128 xmm1 {_mm256_extractf128_ps(*acc, 1)};
                const __m128 xmm2 {_mm_add_ps(xmm0, xmm1)};
                __m128 xmm3 {_mm_movehdup_ps(xmm2)};
                __m128 xmm4 {_mm_add_ps(xmm2, xmm3)};
                xmm3 = _mm_movehl_ps(xmm3, xmm4);
                xmm4 = _mm_add_ss(xmm4, xmm3);
                sum = _mm_cvtss_f32(xmm4);
            #endif
            for (std::int64_t i{k}; i < n; ++i) { // Process leftovers scalar-wise
                sum += x[i]*y[i];
            }
            return sum;
        #elif defined(__SSE2__)
            constexpr std::int64_t step {16};
            const std::int64_t k {n & -step};
            __m128 acc[4] {_mm_setzero_ps()};
            __m128 vx[4];
            __m128 vy[4];
            for (std::int64_t i {}; i < k; i += step) {
                #pragma GCC unroll 4
                for (std::int64_t j {}; j < 4; ++j) {
                    vx[j] = _mm_loadu_ps(x+i+(j<<2));
                    vy[j] = _mm_loadu_ps(y+i+(j<<2));
                    acc[j] = _mm_add_ps(acc[j], _mm_mul_ps(vx[j], vy[j]));
                }
            }
            acc[1] = _mm_add_ps(acc[1], acc[3]);
            *acc = _mm_add_ps(*acc, acc[2]);
            *acc = _mm_add_ps(*acc, acc[1]);
            float sum;
            #if defined(PT_X86_X64_USE_HADD) && defined(__SSE3__)
                *acc = _mm_hadd_ps(*acc, *acc);
                *acc = _mm_hadd_ps(*acc, *acc);
                sum = _mm_cvtss_f32(*acc);
            #else
                __m128 shuf {_mm_shuffle_ps(*acc, *acc, _MM_SHUFFLE(2, 3, 0, 1))};
                __m128 sums {_mm_add_ps(*acc, shuf)};
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                sum = _mm_cvtss_f32(sums);
            #endif
            for (std::int64_t i{k}; i < n; ++i) { // Process leftovers scalar-wise
                sum += x[i]*y[i];
            }
            return sum;
        #elif defined(__ARM_NEON)
            constexpr std::int64_t step {16};
            const std::int64_t k {n & -step};
            float32x4_t acc[4] {vdupq_n_f32(0)};
            float32x4_t vx[4]; // NOLINT(*-pro-type-member-init)
            float32x4_t vy[4]; // NOLINT(*-pro-type-member-init)
            for (std::int64_t i {}; i < k; i += step) { // Vectorize
                #pragma GCC unroll 4
                for (std::int64_t j {}; j < 4; ++j) { // Unroll
                    vx[j] = vld1q_f32(x+i+(j<<2));
                    vy[j] = vld1q_f32(y+i+(j<<2));
                    acc[j] = vfmaq_f32(acc[j], vx[j], vy[j]); // Fused multiply-accumulate
                }
            }
            acc[1] = vaddq_f32(acc[1], acc[3]); // Reduce to scalar with horizontal sum
            *acc = vaddq_f32(*acc, acc[2]); // Reduce to scalar with horizontal sum
            *acc = vaddq_f32(*acc, acc[1]); // Reduce to scalar with horizontal sum
            float sum {vaddvq_f32(*acc)}; // Reduce to scalar with horizontal sum
            for (std::int64_t i {k}; i < n; ++i) { // Process leftovers scalar-wise
                sum += x[i]*y[i];
            }
            return sum;
        #else
            double sum {}; // Higher precision accumulator
            for (std::int64_t i {}; i < n; ++i) {
                sum += static_cast<double>(x[i]*y[i]);
            }
            return static_cast<float>(sum);
        #endif
        }
    }
}
