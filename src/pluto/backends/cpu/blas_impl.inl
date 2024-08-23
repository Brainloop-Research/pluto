// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "../../tensor.hpp"
#include "../../f16.hpp"
#include "../../bf16.hpp"

#include <array>
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <numbers>

#ifdef __ARM_NEON
#   include <arm_neon.h>
#endif
#if defined(_MSC_VER) || defined(__MINGW32__)
#   include <intrin.h>
#elif defined(__x86_64__) || defined(_M_AMD64)
#   include <immintrin.h>
#   define PT_X86_X64_USE_HADD // Prefer horizontal sum with haddps/vhaddps over manual sum
#endif

namespace pluto::backends::cpu::blas {
    static constexpr float sqrt2pi {0.79788456080286535587989211986876f}; // √(2/π)
    static constexpr float gelu_coeff {0.044715f}; // GeLU coefficient

    // Convert scalar f16 to f32
    [[nodiscard]] static auto s_cvt_f16_to_f32(const f16 x) noexcept -> float {
        #if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
            return static_cast<float>(std::bit_cast<__fp16>(x));
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

    // Convert scalar f32 to f16
    [[nodiscard]] static auto s_cvt_f32_to_f16(const float x) noexcept -> f16 {
        std::uint16_t bits;
        #if defined(__ARM_NEON) && !defined(_MSC_VER) // Fast hardware path
            const __fp16 ff16 {static_cast<__fp16>(x)};
            bits = std::bit_cast<std::uint16_t>(ff16);
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
        return f16{bits};
    }

    auto v_cvt_f16_to_f32(const dim n, float* const o, const f16* const x) noexcept -> void {
        if (n == 0) return;
        if (n == 1) {
            *o = s_cvt_f16_to_f32(*x);
            return;
        }
        dim i {};
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
            o[i] = s_cvt_f16_to_f32(x[i]);
        }
    }

    auto v_cvt_f32_to_f16(const dim n, f16* const o, const float* const x) noexcept -> void {
        if (n == 0) return;
        if (n == 1) {
            *o = s_cvt_f32_to_f16(*x);
            return;
        }
        dim i {};
        #ifdef __F16C__
            for (; i+7 < n; i += 8) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(o+i), _mm256_cvtps_ph( _mm256_loadu_ps(x+i), _MM_FROUND_TO_NEAREST_INT));
            }
            for(; i+3 < n; i += 4) {
                _mm_storel_epi64(reinterpret_cast<__m128i*>(o+i),_mm_cvtps_ph(_mm_loadu_ps(x+i),_MM_FROUND_TO_NEAREST_INT));
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
            o[i] = s_cvt_f32_to_f16(x[i]);
        }
    }

    // Convert scalar bf16 to f32
    [[nodiscard]] static auto s_cvt_bf16_to_f32(const bf16 x) noexcept -> float {
        return std::bit_cast<float>(static_cast<std::uint32_t>(x.bits) << 16); // bf16 is basically a truncated f32
    }

    // Convert scalar f32 to bf16
    [[nodiscard]] static auto s_cvt_f32_to_bf16(const float x) noexcept -> bf16 {
        std::uint16_t bits {};
        const auto bi {std::bit_cast<std::uint32_t>(x)};
        if ((bi & 0x7fffffff) > 0x7f800000) { // NaN
            bits = 64 | (bi>>16); // quiet NaNs only
        }
        if (!(bi & 0x7f800000)) { // Subnormals
            bits = (bi & 0x80000000)>>16; // Flush to zero
        }
        bits = (bi + (0x7fff + ((bi>>16) & 1)))>>16; // Rounding and composing final bf16 value
        return bf16{bits};
    }

    auto v_cvt_bf16_to_f32(const dim n, float* const o, const bf16* const x) noexcept -> void {
        if (n == 0) return;
        if (n == 1) {
            *o = s_cvt_bf16_to_f32(*x);
            return;
        }
        dim i {};
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
            o[i] = s_cvt_bf16_to_f32(x[i]);
        }
    }

    auto v_cvt_f32_to_bf16(const dim n, bf16* const o, const float* const x) noexcept -> void {
        if (n == 0) return;
        if (n == 1) {
            *o = s_cvt_f32_to_bf16(*x);
            return;
        }
        dim i {};
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
            o[i] = s_cvt_f32_to_bf16(x[i]);
        }
    }


    template <>
    auto PT_HOTPROC v_softmax(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = std::exp(x[i]);
        }
    }

    template <>
    auto PT_HOTPROC v_sigmoid(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = 1.0f / (1.0f + std::exp(-x[i]));
        }
    }

    template <>
    auto PT_HOTPROC v_tanh(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = std::tanh(x[i]);
        }
    }

    template <>
    auto PT_HOTPROC v_relu(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = std::max(x[i], 0.0f);
        }
    }

    template <>
    auto PT_HOTPROC v_gelu(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = 0.5f * x[i] * (1.0f + std::tanh(sqrt2pi * x[i] * (1.0f + gelu_coeff * x[i] * x[i])));
        }
    }

    template <>
    auto PT_HOTPROC v_silu(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    template <>
    auto PT_HOTPROC v_add(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x,
        const float* __restrict__ const y
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = x[i] + y[i];
        }
    }

    template <>
    auto PT_HOTPROC v_sub(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x,
        const float* __restrict__ const y
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = x[i] - y[i];
        }
    }

    template <>
    auto PT_HOTPROC v_mul(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x,
        const float* __restrict__ const y
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = x[i] * y[i];
        }
    }

    template <>
    auto PT_HOTPROC v_div(
        const dim n,
        float* __restrict__ const o,
        const float* __restrict__ const x,
        const float* __restrict__ const y
    ) noexcept -> void {
        for (dim i {}; i < n; ++i) {
            o[i] = x[i] / y[i];
        }
    }

    template <>
    auto PT_HOTPROC v_dot(
        const dim n,
        const float* __restrict__ const x,
        const float* __restrict__ const y
    ) noexcept -> float {
    #ifdef __AVX512F__
        constexpr dim step {64};
        const dim k {n & -step};
        __m512 acc[4] = {_mm512_setzero_ps()};
        __m512 vx[4];
        __m512 vy[4];
        for (dim i {}; i < k; i += step) {
            #pragma GCC unroll 4
            for (dim j {}; j < 4; ++j) {
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
        constexpr dim step {32};
        const dim k {n & -step};
        __m256 acc[4] {_mm256_setzero_ps()};
        __m256 vx[4];
        __m256 vy[4];
        for (dim i {}; i < k; i += step) {
            #pragma GCC unroll 4
            for (dim j {}; j < 4; ++j) {
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
        for (dim i{k}; i < n; ++i) { // Process leftovers scalar-wise
            sum += x[i]*y[i];
        }
        return sum;
    #elif defined(__SSE2__)
        constexpr dim step {16};
        const dim k {n & -step};
        __m128 acc[4] {_mm_setzero_ps()};
        __m128 vx[4];
        __m128 vy[4];
        for (dim i {}; i < k; i += step) {
            #pragma GCC unroll 4
            for (dim j {}; j < 4; ++j) {
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
        for (dim i{k}; i < n; ++i) { // Process leftovers scalar-wise
            sum += x[i]*y[i];
        }
        return sum;
    #elif defined(__ARM_NEON)
        constexpr dim step {16};
        const dim k {n & -step};
        float32x4_t acc[4] {vdupq_n_f32(0)};
        float32x4_t vx[4]; // NOLINT(*-pro-type-member-init)
        float32x4_t vy[4]; // NOLINT(*-pro-type-member-init)
        for (dim i {}; i < k; i += step) { // Vectorize
            #pragma GCC unroll 4
            for (dim j {}; j < 4; ++j) { // Unroll
                vx[j] = vld1q_f32(x+i+(j<<2));
                vy[j] = vld1q_f32(y+i+(j<<2));
                acc[j] = vfmaq_f32(acc[j], vx[j], vy[j]); // Fused multiply-accumulate
            }
        }
        acc[1] = vaddq_f32(acc[1], acc[3]); // Reduce to scalar with horizontal sum
        *acc = vaddq_f32(*acc, acc[2]); // Reduce to scalar with horizontal sum
        *acc = vaddq_f32(*acc, acc[1]); // Reduce to scalar with horizontal sum
        float sum {vaddvq_f32(*acc)}; // Reduce to scalar with horizontal sum
        for (dim i {k}; i < n; ++i) { // Process leftovers scalar-wise
            sum += x[i]*y[i];
        }
        return sum;
    #else
        double sum {}; // Higher precision accumulator
        for (dim i {}; i < n; ++i) {
            sum += static_cast<double>(x[i]*y[i]);
        }
        return static_cast<float>(sum);
    #endif
    }

    namespace detail {
        template <typename F, typename S>
        concept is_vector_op = requires {
            is_dtype<S>;
            std::is_nothrow_invocable_r_v<void, F, S*, const S*, const S*>; // auto f(S* r, const S* x, const S* y) -> void
        };

        template <typename F, typename S>
        concept is_scalar_op = requires {
            is_dtype<S>;
            std::is_nothrow_invocable_r_v<S, F, S, S>; // auto f(S x, S y) -> S
        };

        template <typename T, typename V_OP> requires requires {
            is_dtype<T>;
            is_vector_op<V_OP, T>;
        }
        static auto PT_AINLINE PT_HOTPROC gen_unary_op(
            [[maybe_unused]] const compute_ctx& ctx,
            tensor& r,          // result
            const tensor& x,    // X = src 0
            V_OP&& v_op         // Vector OP
        ) noexcept -> void {
            assert(x.is_shape_eq(&r));  // Debug only verification - ! must be checked by validation function, TODO: Check broadcasting OP
            auto* const b_r{reinterpret_cast<std::byte*>(r.buf().data())};                                            // Data base ptr
            const auto* const b_x{reinterpret_cast<const std::byte*>(x.buf().data())};                           // Data base ptr
            const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};          // Strides of x
            const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};          // Strides of r
            const dim rc {r.row_count()};
            const dim cc {r.col_count()};
            for (dim row {}; row < rc; ++row) {
                std::invoke(
                    v_op,
                    cc,
                    reinterpret_cast<T*>(b_r + row*r_s1),
                    reinterpret_cast<const T*>(b_x + row*x_s1)
                );
            }
        }

        template <typename T, typename V_OP, typename S_OP> requires requires {
            is_dtype<T>;
            is_vector_op<V_OP, T>;
            is_scalar_op<S_OP, T>;
        }
        static auto PT_AINLINE PT_HOTPROC gen_binary_op(
            const compute_ctx& ctx,
            tensor& r,          // result
            const tensor& x,    // X = src 0
            const tensor& y,    // Y = src 1
            V_OP&& v_op,        // Vector OP
            S_OP&& s_op         // Scalar OP
        ) noexcept -> void {
            assert(x.is_shape_eq(&r));  // Debug only verification - ! must be checked by validation function, TODO: Check broadcasting OP
            auto* const b_r{reinterpret_cast<std::byte*>(r.buf().data())};                    // Data base ptr
            const auto* const b_x{reinterpret_cast<const std::byte*>(x.buf().data())};   // Data base ptr
            const auto* const b_y{reinterpret_cast<const std::byte*>(y.buf().data())};   // Data base ptr
            const auto [x_d0, x_d1, x_d2, x_d3] {x.shape()};   // Dimensions of x
            const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()}; // Strides of x
            const auto [y_d0, y_d1, y_d2, y_d3] {y.shape()};   // Dimensions of y
            const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()}; // Strides of y
            const auto [r_d0, r_d1, r_d2, r_d3] {r.shape()};   // Dimensions of r
            const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()}; // Strides of r
            const dim rc {r.row_count()};                                   // Row count (number of columns in first dim): r.dims()[0]
            const dim tidx {ctx.thread_idx};                                // Current thread index
            const dim tc {ctx.num_threads};                                 // Current thread count
            const dim rpt {(rc + tc - 1)/tc};                               // Rows per thread
            const dim row_start {rpt * tidx};                               // Current thread row interval start
            const dim row_end {std::min(row_start + rpt, rc)};              // Current thread row interval end
            if (sizeof(T) == y.strides().front()) {                         // Fast path - dense kernel for contiguous layout
                for (dim row_i {row_start}; row_i < row_end; ++row_i) {     // For each row
                    const dim x_i3 {row_i / (x_d2*x_d1)};                   // Dimension 3 - Linear to multidim index
                    const dim x_i2 {(row_i - x_i3*x_d2*x_d1)/x_d1};         // Dimension 2 - Linear to multidim index
                    const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};    // Dimension 1 - Linear to multidim index
                    const dim y_i3 {x_i3 % y_d3};                           // Dimension 3 Broadcast x -> y
                    const dim y_i2 {x_i2 % y_d2};                           // Dimension 2 Broadcast x -> y
                    const dim y_i1 {x_i1 % y_d1};                           // Dimension 1 Broadcast x -> y
                    auto* const p_r {reinterpret_cast<T*>(b_r + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                    const auto* const p_x {reinterpret_cast<const T*>(b_x + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                    const auto* const p_y {reinterpret_cast<const T*>(b_y + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1)};
                    for (dim i {}; i < x_d0 / y_d0; ++i) { // Macro kernel
                        std::invoke(v_op, y_d0, p_r + i*y_d0, p_x + i*y_d0, p_y); // Micro Kernel -> apply vector operation
                    }
                }
            } else {
                for (dim row_i {row_start}; row_i < row_end; ++row_i) {          // For each row
                    const dim x_i3 {row_i / (x_d2*x_d1)};                        // Dimension 3 - Linear to multidim index
                    const dim x_i2 {(row_i - x_i3*x_d2*x_d1)/x_d1};              // Dimension 2 - Linear to multidim index
                    const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};         // Dimension 1 - Linear to multidim index
                    const dim y_i3 {x_i3 % y_d3};                                // Dimension 3 Broadcast x -> y
                    const dim y_i2 {x_i2 % y_d2};                                // Dimension 2 Broadcast x -> y
                    const dim y_i1 {x_i1 % y_d1};                                // Dimension 1 Broadcast x -> y
                    auto* const p_r {reinterpret_cast<T*>(b_r + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                    const auto* const p_x {reinterpret_cast<const T*>(b_x + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                    for (dim i {}; i < r_d0; ++i) { // Micro kernel
                        const auto* const p_y {reinterpret_cast<const T*>(b_y + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i%y_d0*y_s0)};
                        p_r[i] = std::invoke(s_op, p_x[i], *p_y); // Apply scalar operation
                    }
                }
            }
        }

        /*
        * BLAS SGEMM (Single precision General Matrix Multiply)
        * Compute the matrix product of two matrices X and Y: R = X @ Y
        * TODO: This is a naive implementation and not optimized.
        * TODO: Thread partitioning
        * TODO: optimize for cache efficiency and SIMD (use vec::dot)
        * TODO: Handle broadcasting
        */
        template <typename T> requires is_dtype<T>
        auto PT_AINLINE PT_HOTPROC gen_gemm(
            const compute_ctx& ctx,
            tensor& r,          // result
            const tensor& x,    // X = src 0
            const tensor& y     // Y = src 1
        ) noexcept -> void;

        template <>
        auto PT_AINLINE PT_HOTPROC gen_gemm<float>( // Compute R = X @ Y
            [[maybe_unused]] const compute_ctx& ctx,
            tensor& r,
            const tensor& x,
            const tensor& y
        ) noexcept -> void {
            assert(x.is_matmul_compatible(&y));
            auto* const b_r {reinterpret_cast<std::byte*>(r.buf().data())};
            const auto* const b_x {reinterpret_cast<const std::byte*>(r.buf().data())};
            const auto* const b_y {reinterpret_cast<const std::byte*>(r.buf().data())};
            const auto [x_d0, x_d1, x_d2, x_d3] {x.shape()};
            const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};
            const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};
            const auto [r_d0, r_d1, r_d2, r_d3] {r.shape()};
            const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};
            for (dim i3 {}; i3 < r_d3; ++i3) {
                for (dim i2 {}; i2 < r_d2; ++i2) {
                    for (dim i1 {}; i1 < r_d1; ++i1) {
                        for (dim i0 {}; i0 < r_d1; ++i0) {
                            double sum {}; // TODO: optimize and use vec::dot
                            for (dim k {}; k < x_d0; ++k) {
                                const auto* const p_x {reinterpret_cast<const float*>(
                                    b_x + k*x_s0 + i0*x_s1 + i2*x_s2 + i3*x_s3
                                )};
                                const auto* const p_y {reinterpret_cast<const float*>(
                                    b_y + i1*y_s0 + k*y_s1 + i2*y_s2 + i3*y_s3
                                )};
                                sum += static_cast<double>(*p_x**p_y);
                            }
                            auto* const p_r {reinterpret_cast<float*>(
                                b_r + i1*r_s0 + i0*r_s1 + i2*r_s2 + i3*r_s3
                            )};
                            *p_r = static_cast<float>(sum);
                        }
                    }
                }
            }
        }
    }

    auto t_softmax(const compute_ctx& ctx, const tensor& x) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_unary_op<float>(ctx, *r, x, v_softmax<float>);
        return r;
    }

    auto t_sigmoid(const compute_ctx& ctx, const tensor& x) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_unary_op<float>(ctx, *r, x, v_sigmoid<float>);
        return r;
    }

    auto t_tanh(const compute_ctx& ctx, const tensor& x) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_unary_op<float>(ctx, *r, x, v_tanh<float>);
        return r;
    }

    auto t_relu(const compute_ctx& ctx, const tensor& x) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_unary_op<float>(ctx, *r, x, v_relu<float>);
        return r;
    }

    auto t_gelu(const compute_ctx& ctx, const tensor& x) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_unary_op<float>(ctx, *r, x, v_gelu<float>);
        return r;
    }

    auto t_silu(const compute_ctx& ctx, const tensor& x) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_unary_op<float>(ctx, *r, x, v_silu<float>);
        return r;
    }

    auto t_add(
        const compute_ctx& ctx,
        const tensor& x,
        const tensor& y
    ) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_binary_op<float>(ctx, *r, x, y, v_add<float>, std::plus<float>{});
        return r;
    }

    auto t_sub(
        const compute_ctx& ctx,
        const tensor& x,
        const tensor& y
    ) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_binary_op<float>(ctx, *r, x, y, v_sub<float>, std::minus<float>{});
        return r;
    }

    auto t_mul(
        const compute_ctx& ctx,
        const tensor& x,
        const tensor& y
    ) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_binary_op<float>(ctx, *r, x, y, v_mul<float>, std::multiplies<float>{});
        return r;
    }

    auto t_div(
        const compute_ctx& ctx,
        const tensor& x,
        const tensor& y
    ) noexcept -> tensor* {
        tensor* const r {x.isomorphic_clone()};
        detail::gen_binary_op<float>(ctx, *r, x, y, v_div<float>, std::divides<float>{});
        return r;
    }

    auto t_matmul(
        const compute_ctx& ctx,
        const tensor& x,
        const tensor& y
    ) noexcept -> tensor* {
        const std::array<dim, max_dims> shape {x.shape()[1], y.shape()[1], y.shape()[2], y.shape()[3]};
        tensor* const r {tensor::create(x.ctx(), shape)};
        detail::gen_gemm<float>(ctx, *r, x, y);
        return r;
    }
}
