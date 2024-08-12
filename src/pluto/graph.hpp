// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#pragma once

namespace pluto {
    constexpr std::size_t max_args {2};

    #define PT_ENUM_SEP ,
    #define pt_opdef(_, __) /* Operator function "ψ" -> Enumerator | Mnemonic | Info | ArgCount <= PT_OP_ARGMAX */ \
    /* Nullary operations ψ(_) (argument unused but same signature as unary) */\
    _(nop, "nop", "!", 1)__\
     /* Unary operations ψ(x) */\
    _(softmax, "softmax", "softmax", 1)__\
    _(sigmoid, "sigmoid", "sigmoid", 1)__\
    _(tanh, "tanh", "tanh", 1)__\
    _(relu, "relu", "relu", 1)__\
    _(gelu, "gelu", "gelu", 1)__\
    _(silu, "silu", "silu", 1)__\
    /* Binary operations ψ(x,y) */\
    _(add, "add", "+", 2)__\
    _(sub, "sub", "-", 2)__\
    _(mul, "mul", "*", 2)__\
    _(div, "div", "/", 2)__\
    _(matmul, "matmul", "@", 2)

    enum class opcode : std::uint32_t {
        #define inject_enum(opc, _, __, ___) opc
        pt_opdef(inject_enum, PT_ENUM_SEP)
        #undef inject_enum
        , len_
    };
    constexpr std::array<std::string_view, static_cast<std::size_t>(opcode::len_)> opcode_names {
        #define inject_enum(_, name, __, ___) name
            pt_opdef(inject_enum, PT_ENUM_SEP)
        #undef inject_enum
    };
    constexpr std::array<std::string_view, static_cast<std::size_t>(opcode::len_)> opcode_mnemonics {
        #define inject_enum(_, __, mnemonic, ___) mnemonic
            pt_opdef(inject_enum, PT_ENUM_SEP)
        #undef inject_enum
    };
    constexpr std::array<std::uint8_t, static_cast<std::size_t>(opcode::len_)> opcode_arg_counts {
        #define inject_enum(_, __, ___, argcount) (argcount)
            pt_opdef(inject_enum, PT_ENUM_SEP)
        #undef inject_enum
    };
    static_assert(std::all_of(opcode_arg_counts.begin(), opcode_arg_counts.end(), [](const std::uint8_t arg) noexcept -> bool { return arg <= max_args; }));

    #undef pt_opdef
    #undef PT_ENUM_SEP

    enum class graph_eval_order : bool {
        left_to_right = true,
        right_to_left = false
    };
}
