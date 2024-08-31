// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <cstdint>
#include <type_traits>

namespace pluto {
    template <typename T>
    concept is_pool_obj = requires {
        std::is_trivially_destructible_v<T>;
    };

    // Thin smart pointer wrapper for objects allocated within the linear bump pointer allocator.
    template <typename T> requires (is_pool_obj<T> && std::negation_v<std::is_void<T>>)
    struct pool_ref {
        constexpr pool_ref() noexcept = default;
        constexpr pool_ref(std::nullptr_t) noexcept {}
        pool_ref(const pool_ref&) noexcept = default;
        pool_ref(pool_ref&&) noexcept = default;
        auto operator = (const pool_ref&) noexcept -> pool_ref& = default;
        auto operator = (pool_ref&&) noexcept -> pool_ref& = default;
        ~pool_ref() = default;
        constexpr auto operator * () noexcept -> T& { return *m_ptr; }
        constexpr auto operator * () const noexcept -> const T& { return *m_ptr; }
        constexpr auto operator -> () noexcept -> T* { return m_ptr; }
        constexpr auto operator -> () const noexcept -> const T* { return m_ptr; }
        constexpr auto operator == (const pool_ref other) const noexcept { return m_ptr == other.m_ptr; }
        constexpr auto operator != (const pool_ref other) const noexcept { return !(*this == other); }
        constexpr auto operator == (const T* const other) const noexcept { return m_ptr == other; }
        constexpr auto operator != (const T* const other) const noexcept { return !(*this == other); }
        constexpr auto operator == (std::nullptr_t) const noexcept { return m_ptr == nullptr; }
        constexpr auto operator != (std::nullptr_t) const noexcept { return !(*this == nullptr); }
        constexpr operator bool () const noexcept { return m_ptr; }

    private:
        constexpr explicit pool_ref(T* const ptr) noexcept : m_ptr{ptr} {}
        T* m_ptr {};
        friend class context;
    };
    static_assert(sizeof(pool_ref<char>) == sizeof(void*));
    static_assert(alignof(pool_ref<char>) == alignof(void*));
}
