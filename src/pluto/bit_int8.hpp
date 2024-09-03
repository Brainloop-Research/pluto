// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "core.hpp"

namespace pluto {
    template <const std::uint32_t n_bits, typename storage> requires requires {
        n_bits > 0;
        n_bits <= 8;
        std::disjunction_v<
            std::is_same<storage, std::uint8_t>,
            std::is_same<storage, std::int8_t>
        >;
    }
    struct bit_int8 final {
        using underlying_type = storage;
        using u_storage = std::make_unsigned<storage>;
        using s_storage = std::make_signed<storage>;
        [[nodiscard]] static constexpr auto mask(const storage x) noexcept -> storage {
            return static_cast<u_storage>(static_cast<u_storage>(x) << (storage_bits - bits)) >> (storage_bits - bits);
        }
        [[nodiscard]] static constexpr auto full_width(const storage x) noexcept -> storage {
            return static_cast<storage>(x << (storage_bits - bits)) >> (storage_bits - bits);
        }
        constexpr bit_int8() noexcept = default;
        constexpr explicit bit_int8(const storage x) noexcept : m_x{mask(x)} {}
        constexpr bit_int8(const bit_int8&) noexcept = default;
        constexpr bit_int8(bit_int8&&) noexcept = default;
        auto operator=(const bit_int8&) noexcept -> bit_int8& = default;
        auto operator=(bit_int8&&) noexcept -> bit_int8& = default;
        static constexpr std::uint32_t bits {n_bits};
        static constexpr std::uint32_t storage_bits {8*sizeof(storage)};
        static constexpr std::uint32_t digits = std::is_signed_v<storage> ? n_bits-1 : n_bits;
        [[nodiscard]] static constexpr auto max() noexcept -> bit_int8 { return bit_int8{(1<<digits) - 1}; }
        [[nodiscard]] static constexpr auto min() noexcept -> bit_int8 {
            return std::is_signed_v<storage> ? bit_int8{1}<<digits : bit_int8{};
        }
        constexpr auto operator*() const noexcept -> storage { return full_width(m_x); }
        constexpr auto operator==(const bit_int8 rhs) const noexcept -> bool { return mask(m_x) == mask(rhs.m_x); }
        constexpr auto operator!=(const bit_int8 rhs) const noexcept -> bool { return !(*this == rhs); }
        constexpr auto operator<(const bit_int8 rhs) const noexcept -> bool { return **this < *rhs; }
        constexpr auto operator>(const bit_int8 rhs) const noexcept -> bool { return **this > *rhs; }
        constexpr auto operator<=(const bit_int8 rhs) const noexcept -> bool { return **this <= *rhs; }
        constexpr auto operator>=(const bit_int8 rhs) const noexcept -> bool { return **this >= *rhs; }
        constexpr auto operator==(const storage rhs) const noexcept -> bool { return mask(m_x) == mask(rhs); }
        constexpr auto operator!=(const storage rhs) const noexcept -> bool { return !(*this == rhs); }
        constexpr auto operator<(const storage rhs) const noexcept -> bool { return **this < *rhs; }
        constexpr auto operator>(const storage rhs) const noexcept -> bool { return **this > *rhs; }
        constexpr auto operator<=(const storage rhs) const noexcept -> bool { return **this <= *rhs; }
        constexpr auto operator>=(const storage rhs) const noexcept -> bool { return **this >= *rhs; }
        constexpr auto operator-() const noexcept -> bit_int8 { return bit_int8{-m_x}; }
        constexpr auto operator~() const noexcept -> bit_int8 { return bit_int8{~m_x}; }
        constexpr auto operator+(const bit_int8 rhs) const noexcept -> bit_int8 { return bit_int8{m_x + rhs.m_x}; }
        constexpr auto operator-(const bit_int8 rhs) const noexcept -> bit_int8 { return bit_int8{m_x - rhs.m_x}; }
        constexpr auto operator*(const bit_int8 rhs) const noexcept -> bit_int8 { return bit_int8{m_x * rhs.m_x}; }
        constexpr auto operator/(const bit_int8 rhs) const noexcept -> bit_int8 { return bit_int8{**this / *rhs.m_x}; }
        constexpr auto operator%(const bit_int8 rhs) const noexcept -> bit_int8 { return bit_int8{**this % *rhs.m_x}; }
        constexpr auto operator&(const bit_int8 rhs) const noexcept -> bit_int8 { return bit_int8{m_x & rhs.m_x}; }
        constexpr auto operator|(const bit_int8 rhs) const noexcept -> bit_int8 { return bit_int8{m_x | rhs.m_x}; }
        constexpr auto operator^(const bit_int8 rhs) const noexcept -> bit_int8 { return bit_int8{m_x ^ rhs.m_x}; }
        constexpr auto operator<<(const std::uint32_t n) const noexcept -> bit_int8 { return bit_int8{m_x << n}; }
        constexpr auto operator>>(const std::uint32_t n) const noexcept -> bit_int8 { return bit_int8{**this >> n}; }
        constexpr auto operator+=(const bit_int8 rhs) noexcept -> bit_int8& { return *this = *this + rhs; }
        constexpr auto operator-=(const bit_int8 rhs) noexcept -> bit_int8& { return *this = *this - rhs; }
        constexpr auto operator*=(const bit_int8 rhs) noexcept -> bit_int8& { return *this = *this * rhs; }
        constexpr auto operator/=(const bit_int8 rhs) noexcept -> bit_int8& { return *this = *this / rhs; }
        constexpr auto operator%=(const bit_int8 rhs) noexcept -> bit_int8& { return *this = *this % rhs; }
        constexpr auto operator&=(const bit_int8 rhs) noexcept -> bit_int8& { return *this = *this & rhs; }
        constexpr auto operator|=(const bit_int8 rhs) noexcept -> bit_int8& { return *this = *this | rhs; }
        constexpr auto operator^=(const bit_int8 rhs) noexcept -> bit_int8& { return *this = *this ^ rhs; }
        constexpr auto operator<<=(const std::uint32_t n) noexcept -> bit_int8& { return *this = *this << n; }
        constexpr auto operator>>=(const std::uint32_t n) noexcept -> bit_int8& { return *this = *this >> n; }

    private:
        storage m_x {};
    };
}
