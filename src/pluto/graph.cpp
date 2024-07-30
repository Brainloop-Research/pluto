// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "blas.hpp"
#include "graph.hpp"

namespace pluto::graph {
    constexpr std::array<verify_op, static_cast<std::size_t>(opcode::len_)> verify_op_lut {
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, [[maybe_unused]] const std::span<tensor*> args) -> bool { // nop
            return true;
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // softmax
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::softmax)] && args[0] != nullptr;
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // sigmoid
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::sigmoid)] && args[0] != nullptr;
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // tanh
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::tanh)] && args[0] != nullptr;
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // relu
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::relu)] && args[0] != nullptr;
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // gelu
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::gelu)] && args[0] != nullptr;
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // silu
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::silu)] && args[0] != nullptr;
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // add
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::add)] && args[0] != nullptr && args[1] != nullptr && args[0] != args[1];
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // sub
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::sub)] && args[0] != nullptr && args[1] != nullptr && args[0] != args[1];
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // mul
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::mul)] && args[0] != nullptr && args[1] != nullptr && args[0] != args[1];
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // div
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::div)] && args[0] != nullptr && args[1] != nullptr && args[0] != args[1];
        },
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, const std::span<tensor*> args) -> bool { // matmul
            return args.size() == opcode_arg_counts[static_cast<std::size_t>(opcode::matmul)] && args[0] != nullptr && args[1] != nullptr && args[0] != args[1];
        }
    };

    constexpr std::array<eval_op, static_cast<std::size_t>(opcode::len_)> eval_op_lut {
        +[]([[maybe_unused]] const blas::compute_ctx& ctx, tensor*& r, [[maybe_unused]] const std::span<tensor*> args) -> bool { // nop
            return r;
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // softmax
            return (r = blas::softmax(ctx, *args[0]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // sigmoid
            return (r = blas::sigmoid(ctx, *args[0]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // tanh
            return (r = blas::tanh(ctx, *args[0]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // relu
            return (r = blas::relu(ctx, *args[0]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // gelu
            return (r = blas::gelu(ctx, *args[0]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // silu
            return (r = blas::silu(ctx, *args[0]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // add
            return (r = blas::add(ctx, *args[0], *args[1]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // sub
            return (r = blas::sub(ctx, *args[0], *args[1]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // mul
            return (r = blas::mul(ctx, *args[0], *args[1]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // div
            return (r = blas::div(ctx, *args[0], *args[1]));
        },
        +[](const blas::compute_ctx& ctx, tensor*& r, const std::span<tensor*> args) -> bool { // matmul
            return (r = blas::mul(ctx, *args[0], *args[1]));
        }
    };

    template <const graph_eval_order Ord, typename F, typename... Args>
    requires std::is_invocable_r_v<bool, F, tensor*, Args...>
    [[nodiscard]] static constexpr auto graph_visit(tensor* const root, F&& f, Args&&... arr)
    noexcept(std::is_nothrow_invocable_r_v<bool, F, tensor*, Args...>) -> bool {
        if (!root) [[unlikely]] return false;
        const std::span<tensor*> args {root->get_args()};
        for (std::size_t i {}, k; i < args.size(); ++i) {
            if constexpr (Ord == graph_eval_order::left_to_right) { k = i; }
            else { k = args.size()-i-1; }
            if (!graph_visit<Ord>(args[k], std::forward<F>(f), std::forward<Args>(arr)...)) [[unlikely]] return false;
        }
        return std::invoke(std::forward<F>(f), root, std::forward<Args>(arr)...);
    }

    auto verify(tensor* const root, const graph_eval_order order) -> bool {
        blas::compute_ctx ctx {};
        const auto verifyer {[&ctx](tensor* const t) -> bool {
            return (*verify_op_lut[static_cast<std::size_t>(t->get_op_code())])(ctx, t->get_args());
        }};
        if (order == graph_eval_order::left_to_right) return graph_visit<graph_eval_order::left_to_right>(root, verifyer);
        else return graph_visit<graph_eval_order::right_to_left>(root, verifyer);
    }

    auto eval(tensor* const root, const graph_eval_order order) -> std::pair<tensor*, bool> {
        blas::compute_ctx ctx {};
        const auto evaluator {[&ctx](tensor* const t, tensor*& r) -> bool {
            return (*eval_op_lut[static_cast<std::size_t>(t->get_op_code())])(ctx, r, t->get_args());
        }};
        std::pair<tensor*, bool> r {};
        if (order == graph_eval_order::left_to_right) r.second = graph_visit<graph_eval_order::left_to_right>(root, evaluator, r.first);
        else r.second = graph_visit<graph_eval_order::right_to_left>(root, evaluator, r.first);
        return r;
    }
}
