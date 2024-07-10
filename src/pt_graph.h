// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#ifndef PT_GRAPH_H
#define PT_GRAPH_H

#include "pt_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PT_OP_ARGMAX 2
#define pt_opdef(_, __) /* ψ Enumerator | Mnemonic | Info | ArgCount <= PT_OP_ARGMAX */ \
    /* Nullary operations ψ(_) (argument unused but same signature as unary) */\
    _(PT_OPC_NOP, "nop", "!", 1)__\
     /* Unary operations ψ(Tx) */\
    _(PT_OPC_SOFTMAX, "softmax", "softmax", 1)__\
    _(PT_OPC_SIGMOID, "sigmoid", "sigmoid", 1)__\
    _(PT_OPC_RELU, "relu", "relu", 1)__\
    /* Binary operations ψ(Tx,Ty) */\
    _(PT_OPC_ADD, "add", "+", 2)__\
    _(PT_OPC_SUB, "sub", "-", 2)__\
    _(PT_OPC_MUL, "mul", "*", 2)__\
    _(PT_OPC_DIV, "div", "/", 2)__\
    _(PT_OPC_MATMUL, "matmul", "@", 2)

enum pt_opcode_t {
#define inject_enum(opc, _, __, ___) opc
    pt_opdef(inject_enum, PT_ENUM_SEP)
#undef inject_enum
    , PT_OPC_MAX
};

struct pt_tensor_t;

typedef bool (*pt_verify_op_t)(
    const struct pt_ctx_t *ctx,
    const struct pt_tensor_t **args,
    size_t n_args
);
typedef struct pt_tensor_t *(*pt_eval_op_t)(
    const struct pt_ctx_t *ctx,
    struct pt_tensor_t **args,
    size_t n_args
);

extern PT_EXPORT const char *const pt_opcode_mnemonic[PT_OPC_MAX]; // Mnemonic of each operation
extern PT_EXPORT const char *const pt_opcode_desc[PT_OPC_MAX]; // Description of each operation
extern PT_EXPORT const uint8_t pt_opcode_arg_count[PT_OPC_MAX]; // Number of arguments for each operation
extern PT_EXPORT const pt_verify_op_t pt_verify_op[PT_OPC_MAX]; // Lookup table for verification functions
extern PT_EXPORT const pt_eval_op_t pt_eval_op[PT_OPC_MAX]; // Lookup table for evaluation functions

#ifdef __cplusplus
}
#endif

#endif
