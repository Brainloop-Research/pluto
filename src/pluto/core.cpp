// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#include "core.hpp"

#include <cstdio>
#include <iomanip>
#include <cassert>
#include <limits>

namespace pluto {
    context::context(
        const std::size_t chunk_size,
        const std::size_t chunk_cap
    ) : m_chunk_size {chunk_size ? chunk_size : k_default_chunk_size} {
        m_chunks.reserve(chunk_cap ? chunk_cap : k_default_chunk_cap);
        push_chunk();
    }

    context::~context() {

    }

    auto context::pool_alloc_raw(const std::size_t size) -> void* {
        assert(size && size <= std::numeric_limits<std::ptrdiff_t>::max());
        if (m_delta - m_chunks.back().get() < static_cast<std::ptrdiff_t>(size)) {
            if (m_chunk_size < size) { // Increase the chunk size if it's too small to accommodate the requested length
                while ((m_chunk_size <<= 1) < size
                    && m_chunk_size <= (std::numeric_limits<std::ptrdiff_t>::max() >> 1));
            }
            push_chunk();
            if constexpr (k_enable_pool_memory_logging) {
                std::fprintf(
                    stderr,
                    "Pool chunk exhausted - requested %.03f KiB\n"
                    "Increase pool chunk size for best performance, current pool chunk size: %.03f MiB, total allocated: %.03f MiB\n",
                    static_cast<double>(size)/static_cast<double>(1<<10),
                    static_cast<double>(m_chunk_size)/static_cast<double>(1<<20),
                    static_cast<double>(m_chunk_size*m_chunks.size())/static_cast<double>(1<<20)
                );
            }
        }
        m_delta -= size;
        ++m_alloc_acc;
        m_alloc_total += size;
        return m_delta;
    }

    auto context::pool_alloc_raw_aligned(std::size_t size, std::size_t align) -> void* {
        assert(align && !(align & (align - 1))); // Alignment must be a power of 2
        const std::size_t a_mask {align - 1};
        return reinterpret_cast<void*>(
            reinterpret_cast<std::uintptr_t>(
                    pool_alloc_raw(size + a_mask)
            )+a_mask & ~a_mask
        );
    }

    auto context::push_chunk() -> void {
        auto chunk {std::make_unique<std::byte[]>(m_chunk_size)};
        m_mapped_total += m_chunk_size;
        m_delta = chunk.get() + m_chunk_size;
        m_chunks.emplace_back(std::move(chunk));
    }
}
