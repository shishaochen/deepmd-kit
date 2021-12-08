#!/usr/bin/env python3
"""
Gradients for prod virial.
"""

from tensorflow.python.framework import ops
from deepmd.env import op_grads_module

@ops.RegisterGradient("ProdForceNb")
def _prod_force_nb_grad_cc(op, grad):
    net_grad = op_grads_module.prod_force_nb_grad(grad,
                                                  op.inputs[1],
                                                  op.inputs[2])
    return [net_grad, None, None]

@ops.RegisterGradient("ProdVirialNb")
def _prod_virial_nb_grad_cc(op, grad, grad_atom):
    # Suppose `grad_atom` is not used
    net_grad = op_grads_module.prod_virial_nb_grad(grad,
                                                   op.inputs[1],
                                                   op.inputs[2],
                                                   op.inputs[3])
    return [net_grad, None, None, None]
