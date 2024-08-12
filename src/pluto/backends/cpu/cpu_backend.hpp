// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include "../../backend.hpp"

namespace pluto::backends {
    class cpu_backend : public backend_interface {
    public:
        cpu_backend();
        ~cpu_backend() override = default;
    };
}
