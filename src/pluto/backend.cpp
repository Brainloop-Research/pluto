// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "backend.hpp"
#include "tensor.hpp"

#include <atomic>

namespace pluto {
    static constinit std::atomic_uint32_t backend_id {0};

    backend_interface::backend_interface(std::string&& name)
        : m_id{backend_id.fetch_add(1, std::memory_order_relaxed)},
        m_name{std::move(name)},
        m_verify_dispatch_table {
            &backend_interface::verify_softmax,
            &backend_interface::verify_sigmoid,
            &backend_interface::verify_tanh,
            &backend_interface::verify_relu,
            &backend_interface::verify_gelu,
            &backend_interface::verify_silu,
            &backend_interface::verify_add,
            &backend_interface::verify_sub,
            &backend_interface::verify_mul,
            &backend_interface::verify_div,
            &backend_interface::verify_matmul
        },
        m_eval_dispatch_table {
            &backend_interface::eval_softmax,
            &backend_interface::eval_sigmoid,
            &backend_interface::eval_tanh,
            &backend_interface::eval_relu,
            &backend_interface::eval_gelu,
            &backend_interface::eval_silu,
            &backend_interface::eval_add,
            &backend_interface::eval_sub,
            &backend_interface::eval_mul,
            &backend_interface::eval_div,
            &backend_interface::eval_matmul
        } {

        }

    template <const graph_eval_order Ord, typename F, typename... Args> requires std::is_invocable_r_v<bool, F, const tensor*, Args...>
    static constexpr auto graph_visit(const tensor* const root, F&& f, Args&&... arr) noexcept(std::is_nothrow_invocable_v<F, const tensor*, Args...>) -> bool {
        if (!root) [[unlikely]]
            return false;
        const std::span<const tensor*> args {root->get_args()};
        for (std::size_t i {}, k; i < args.size(); ++i) {
            if constexpr (Ord == graph_eval_order::left_to_right) { k = i; }
            else { k = args.size()-i-1; }
            if (!graph_visit<Ord>(args[k], std::forward<F>(f), std::forward<Args>(arr)...)) [[unlikely]]
                return false;
        }
        return std::invoke(std::forward<F>(f), root, std::forward<Args>(arr)...);
    }

    auto backend_interface::verify(const compute_ctx& ctx, tensor* const root, const graph_eval_order order) -> bool {
        const auto verifyer {[&](const tensor* const t) -> bool {
            const auto opc = static_cast<std::size_t>(t->get_op_code());
            return (this->*m_verify_dispatch_table[opc])(ctx, t->get_args());
        }};
        return order == graph_eval_order::left_to_right
           ? graph_visit<graph_eval_order::left_to_right>(root, verifyer)
           : graph_visit<graph_eval_order::right_to_left>(root, verifyer);
    }

    auto backend_interface::compute(const compute_ctx& ctx, tensor* const root, const graph_eval_order order) -> tensor* {
        const auto evaluator {[&](const tensor* const t, tensor*& r) -> bool {
            const auto opc = static_cast<std::size_t>(t->get_op_code());
            r = (this->*m_eval_dispatch_table[opc])(ctx, t->get_args());
            return true;
        }};
        tensor* r {};
        const bool result {
            order == graph_eval_order::left_to_right
            ? graph_visit<graph_eval_order::left_to_right>(root, evaluator, r)
            : graph_visit<graph_eval_order::right_to_left>(root, evaluator, r)
        };
        assert(result && r);
        return r;
    }

    [[nodiscard]] static auto verify_unary(const std::span<const tensor*> args) noexcept -> bool {
        return args.size() == 1 && args[0] != nullptr;
    }
    [[nodiscard]] static auto verify_binary(const std::span<const tensor*> args) noexcept -> bool {
        return args.size() == 2 && args[0] != nullptr && args[1] != nullptr;
    }

    auto backend_interface::verify_softmax([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_unary(args);
    }

    auto backend_interface::verify_sigmoid([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_unary(args);
    }

    auto backend_interface::verify_tanh([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_unary(args);
    }

    auto backend_interface::verify_relu([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_unary(args);
    }

    auto backend_interface::verify_gelu([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_unary(args);
    }

    auto backend_interface::verify_silu([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_unary(args);
    }

    auto backend_interface::verify_add([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_binary(args);
    }

    auto backend_interface::verify_sub([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_binary(args);
    }

    auto backend_interface::verify_mul([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_binary(args);
    }

    auto backend_interface::verify_div([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_binary(args);
    }

    auto backend_interface::verify_matmul([[maybe_unused]] const compute_ctx& ctx, std::span<const tensor*> args) const noexcept -> bool {
        return verify_binary(args);
    }
}
