// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "backend.hpp"
#include "tensor.hpp"
#include "graph.hpp"

namespace pluto::graph {
    template <const graph_eval_order Ord, typename F, typename... Args>
    requires std::is_invocable_r_v<bool, F, tensor*, Args...>
    static constexpr auto graph_visit(tensor* const root, F&& f, Args&&... arr) noexcept(std::is_nothrow_invocable_v<F, tensor*, Args...>) -> bool {
        if (!root) [[unlikely]]
            return false;
        const std::span<tensor*> args {root->get_args()};
        for (std::size_t i {}, k; i < args.size(); ++i) {
            if constexpr (Ord == graph_eval_order::left_to_right) { k = i; }
            else { k = args.size()-i-1; }
            if (!graph_visit<Ord>(args[k], std::forward<F>(f), std::forward<Args>(arr)...)) [[unlikely]]
                return false;
        }
        return std::invoke(std::forward<F>(f), root, std::forward<Args>(arr)...);
    }

    auto verify(tensor* const root, const graph_eval_order order) -> bool {
        blas::compute_ctx ctx {};
        const auto verifyer {[&ctx](tensor* const t) -> bool {
            return (*verify_op_lut[static_cast<std::size_t>(t->get_op_code())])(ctx, t->get_args());
        }};
        return order == graph_eval_order::left_to_right
            ? graph_visit<graph_eval_order::left_to_right>(root, verifyer)
            : graph_visit<graph_eval_order::right_to_left>(root, verifyer);
    }

    auto eval(tensor* const root, const graph_eval_order order) -> tensor* {
        blas::compute_ctx ctx {};
        const auto evaluator {[&ctx](tensor* const t, tensor*& r) -> bool {
            (*eval_op_lut[static_cast<std::size_t>(t->get_op_code())])(ctx, r, t->get_args());
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
}
