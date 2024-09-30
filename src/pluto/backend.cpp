// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "backend.hpp"
#include "tensor.hpp"

#include <atomic>
#include <iostream>

namespace pluto {
    static constinit std::atomic_uint32_t backend_id {0};

    backend_interface::backend_interface(std::string&& name)
        : m_id{backend_id.fetch_add(1, std::memory_order_relaxed)},
        m_name{std::move(name)},
        m_verify_dispatch_table {
            &backend_interface::verify_nop,
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
            &backend_interface::eval_nop,
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

    template <const graph_eval_order Ord, typename F, typename... Args> requires std::is_invocable_r_v<bool, F, tensor*, Args...>
    static constexpr auto graph_visit(tensor* const root, F&& f, Args&&... arr) noexcept(std::is_nothrow_invocable_v<F, tensor*, Args...>) -> bool {
        if (!root) [[unlikely]] return false;
        if (root->is_leaf_node()) return true;
        const std::span<pool_ref<tensor>> args {root->get_args()};
        for (std::size_t i {}, k; i < args.size(); ++i) {
            if constexpr (Ord == graph_eval_order::left_to_right) { k = i; }
            else { k = args.size()-i-1; }
            if (!graph_visit<Ord>(&*args[k], std::forward<F>(f), std::forward<Args>(arr)...)) [[unlikely]]
                return false;
        }
        return std::invoke(std::forward<F>(f), root, std::forward<Args>(arr)...);
    }

    auto backend_interface::verify(const compute_ctx& ctx, pool_ref<tensor> root, const graph_eval_order order) -> bool {
        const auto verifyer {[&](const tensor* const t) -> bool {
            const auto opc = static_cast<std::size_t>(t->get_op_code());
            return (this->*m_verify_dispatch_table[opc])(ctx, t);
        }};
        return order == graph_eval_order::left_to_right
           ? graph_visit<graph_eval_order::left_to_right>(&*root, verifyer)
           : graph_visit<graph_eval_order::right_to_left>(&*root, verifyer);
    }

    auto backend_interface::compute(const compute_ctx& ctx, pool_ref<tensor> root, const graph_eval_order order) -> pool_ref<tensor> {
        const auto evaluator {[&](tensor* const r) -> bool {
            const auto opc = static_cast<std::size_t>(r->get_op_code());
            (this->*m_eval_dispatch_table[opc])(ctx, r);
            return true;
        }};
        const bool result {
            order == graph_eval_order::left_to_right
            ? graph_visit<graph_eval_order::left_to_right>(&*root, evaluator)
            : graph_visit<graph_eval_order::right_to_left>(&*root, evaluator)
        };
        assert(result);
        return root;
    }

    #define verify_expr(expr) \
        if (!(expr)) [[unlikely]] { \
             std::cerr << "Compute graph verification failed!\nTest failed: " << #expr << std::endl; \
             return false; \
        }

    [[nodiscard]] static auto verify_base(
        const opcode opc,
        const tensor* const node
    ) noexcept -> bool {
        verify_expr(node != nullptr);
        verify_expr(node->get_args().size() == opcode_arg_counts[static_cast<std::size_t>(opc)]);
        for (auto&& arg : node->get_args()) {
            verify_expr(arg != nullptr);
        }
        return true;
    }

    auto backend_interface::verify_nop([[maybe_unused]] const compute_ctx& ctx, [[maybe_unused]] const tensor* const node) const noexcept -> bool {
        return true;
    }

    auto backend_interface::verify_softmax([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::softmax, node);
    }

    auto backend_interface::verify_sigmoid([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::sigmoid, node);
    }

    auto backend_interface::verify_tanh([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::tanh, node);
    }

    auto backend_interface::verify_relu([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::relu, node);
    }

    auto backend_interface::verify_gelu([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::gelu, node);
    }

    auto backend_interface::verify_silu([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::silu, node);
    }

    auto backend_interface::verify_add([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::add, node);
    }

    auto backend_interface::verify_sub([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::sub, node);
    }

    auto backend_interface::verify_mul([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::mul, node);
    }

    auto backend_interface::verify_div([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::div, node);
    }

    auto backend_interface::verify_matmul([[maybe_unused]] const compute_ctx& ctx, const tensor* const node) const noexcept -> bool {
        return verify_base(opcode::matmul, node);
    }

    auto backend_interface::eval_nop([[maybe_unused]] const compute_ctx& ctx, [[maybe_unused]] tensor* node) const noexcept -> void {

    }
}
