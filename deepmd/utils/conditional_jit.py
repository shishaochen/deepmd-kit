import contextlib
import logging
import os

from deepmd.env import tf


@contextlib.contextmanager
def dummy_scope():
  yield


def conditional_jit_scope(use_conditional_jit=None,
                          compile_ops=True, separate_compiled_gradients=False):
  """Create jit scope conditionally.

    If ENABLE_CONDITIONAL_JIT is set, we will use conditional_jit.

  Args:
    use_conditional_jit (bool): If true, use conditional_jit.
    compile_ops (bool): enable xla or not.
    separate_compiled_gradients (bool): If true put each gradient subgraph into
      a separate compilation scope.

  Returns:
    scope

  """
  if use_conditional_jit is None:
    if os.getenv("ENABLE_CONDITIONAL_JIT") == "1":
      use_conditional_jit = True

  if use_conditional_jit:  # not tensorflow.python.eager.context.executing_eagerly()
    logging.warning("Enable conditional JIT ...")
    if hasattr(tf, "xla"):
      return tf.xla.experimental.jit_scope(
        compile_ops=compile_ops,
        separate_compiled_gradients=separate_compiled_gradients)

    # tf <= 1.13
    logging.warning("Please upgrade TensorFlow to 1.15 or higher!")
  return dummy_scope()