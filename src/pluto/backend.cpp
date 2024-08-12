// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "backend.hpp"

#include <atomic>

namespace pluto {
    static constinit std::atomic_uint32_t backend_id {0};

    backend_interface::backend_interface(std::string&& name)
        : m_id{backend_id.fetch_add(1, std::memory_order_relaxed)}, m_name{std::move(name)} {}
}
