// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

// Configuration macros

#define PT_ENABLE_LOGGING // Enable logging
#define PT_EXPORT_DLL // Export functions for shared library usage

// Utility macros

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#	define PT_AINLINE inline __attribute__((always_inline))
#	define PT_NOINLINE __attribute__((noinline))
#   define PT_HOTPROC __attribute__((hot))
#   define PT_COLDPROC __attribute__((cold))
#   define PT_PACKED __attribute__((packed))
#	define pt_likely(x) __builtin_expect(!!(x), 1)
#	define pt_unlikely(x) __builtin_expect(!!(x), 0)
#else
#	define PT_AINLINE inline __forceinline
#	define PT_NOINLINE __declspec(noinline)
#   define PT_HOTPROC
#   define PT_COLDPROC
#   define PT_PACKED __declspec(align(1))
#	define pt_likely(x) (x)
#	define pt_unlikely(x) (x)
#endif

#ifdef PT_EXPORT_DLL
#   ifdef _MSC_VER
#       define PT_EXPORT __declspec(dllexport)
#   else
#       define PT_EXPORT __attribute__((visibility("default")))
#   endif
#else
#   define PT_EXPORT
#endif

#define PT_ENUM_SEP ,

namespace pluto {
    class context final {
    public:
        static constexpr std::size_t k_default_chunk_size {1<<20};
        static constexpr std::size_t k_default_chunk_cap {1<<3};
        static constexpr bool k_enable_pool_memory_logging {false};

        explicit context(
            std::size_t chunk_size = k_default_chunk_size,
            std::size_t chunk_cap = k_default_chunk_cap
        );
        context(const context&) = delete;
        context(context&&) = delete;
        auto operator=(const context&) -> context& = delete;
        auto operator=(context&&) -> context& = delete;
        ~context();

        [[nodiscard]] auto pool_alloc_raw(std::size_t size) -> void*;
        [[nodiscard]] auto pool_alloc_raw_aligned(std::size_t size, std::size_t align) -> void*;

        template <typename T, typename... Args>
            requires std::is_standard_layout_v<T>
                && std::is_trivially_destructible_v<T>
                && std::is_constructible_v<T, Args...>
        [[nodiscard]] auto pool_alloc(Args&&... args) -> T* {
            T* obj;
            if constexpr (alignof(T) <= alignof(std::max_align_t) && !(alignof(T) & (alignof(T)-1))) {
                obj = static_cast<T*>(pool_alloc_raw(sizeof(T)));
            } else { obj = static_cast<T*>(pool_alloc_raw_aligned(sizeof(T), alignof(T))); }
            return std::launder<T>(new(obj) T{std::forward<Args>(args)...});
        }

    private:
        auto push_chunk() -> void;

        std::size_t m_chunk_size {};
        std::vector<std::unique_ptr<std::byte[]>> m_chunks {};
        std::byte* m_delta {};
        std::size_t m_alloc_acc {};
        std::size_t m_mapped_total {};
        std::size_t m_alloc_total {};
    };
}
