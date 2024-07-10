// (c) 2024 Brainloop Research, Mario Sieg. <mario.sieg.64@gmail.com>

#include "pt_graph.h"

#define inject_enum(_, mnemonic, __, ___) mnemonic
const char *const pt_opcode_mnemonic[PT_OPC_MAX] = { pt_opdef(inject_enum, PT_ENUM_SEP) };
#undef inject_enum
#define inject_enum(_, __, desc, ___) desc
const char *const pt_opcode_desc[PT_OPC_MAX] = { pt_opdef(inject_enum, PT_ENUM_SEP) };
#undef inject_enum
#define inject_enum(_, __, ___, nargs) (pt_min(PT_OP_ARGMAX, 255&nargs))
const uint8_t pt_opcode_arg_count[PT_OPC_MAX] = { pt_opdef(inject_enum, PT_ENUM_SEP) };
#undef inject_enum

#define pt_impl_verify_op(opc, body)\
    static bool pt_verify_unary_##opc(\
        const struct pt_ctx_t *const ctx,\
        const struct pt_tensor_t *const x,\
        const struct pt_tensor_t *const y\
    ){\
        (void)ctx, (void)x, (void)y;\
        body\
    }

#define pt_impl_eval_op(opc, body)\
    static struct pt_tensor_t *pt_eval_unary_##opc(\
        const struct pt_ctx_t *const ctx,\
        struct pt_tensor_t *const x,\
        struct pt_tensor_t *const y\
    ){\
        (void)ctx, (void)x, (void)y;\
        body\
    }

pt_impl_verify_op(nop, {
    return true; // NO-OP
})
pt_impl_eval_op(nop, {
    return x; // NO-OP
})

pt_impl_verify_op(softmax, {
    return true; // TODO
})
pt_impl_eval_op(softmax, {
    return x; // TODO
})

pt_impl_verify_op(sigmoid, {
    return true; // TODO
})
pt_impl_eval_op(sigmoid, {
    return x; // TODO
})

pt_impl_verify_op(relu, {
    return true; // TODO
})
pt_impl_eval_op(relu, {
    return x; // TODO
})

pt_impl_verify_op(add, {
    return true; // TODO
})
pt_impl_eval_op(add, {
    return x; // TODO
})

pt_impl_verify_op(sub, {
    return true; // TODO
})
pt_impl_eval_op(sub, {
    return x; // TODO
})

pt_impl_verify_op(mul, {
    return true; // TODO
})
pt_impl_eval_op(mul, {
    return x; // TODO
})

pt_impl_verify_op(div, {
    return true; // TODO
})
pt_impl_eval_op(div, {
    return x; // TODO
})

pt_impl_verify_op(matmul, {
    return true; // TODO
})
pt_impl_eval_op(matmul, {
    return x; // TODO
})

const pt_verify_op_t pt_verify_op[PT_OPC_MAX] = {
        [PT_OPC_NOP] = &pt_verify_unary_nop,
        [PT_OPC_SOFTMAX] = &pt_verify_unary_softmax,
        [PT_OPC_SIGMOID] = &pt_verify_unary_sigmoid,
        [PT_OPC_RELU] = &pt_verify_unary_relu,
        [PT_OPC_ADD] = &pt_verify_unary_add,
        [PT_OPC_SUB] = &pt_verify_unary_sub,
        [PT_OPC_MUL] = &pt_verify_unary_mul,
        [PT_OPC_DIV] = &pt_verify_unary_div,
        [PT_OPC_MATMUL] = &pt_verify_unary_matmul
};
const pt_eval_op_t pt_eval_op[PT_OPC_MAX] = {
        [PT_OPC_NOP] = &pt_eval_unary_nop,
        [PT_OPC_SOFTMAX] = &pt_eval_unary_softmax,
        [PT_OPC_SIGMOID] = &pt_eval_unary_sigmoid,
        [PT_OPC_RELU] = &pt_eval_unary_relu,
        [PT_OPC_ADD] = &pt_eval_unary_add,
        [PT_OPC_SUB] = &pt_eval_unary_sub,
        [PT_OPC_MUL] = &pt_eval_unary_mul,
        [PT_OPC_DIV] = &pt_eval_unary_div,
        [PT_OPC_MATMUL] = &pt_eval_unary_matmul
};
