"""
Shim module that re-exports scikit-learn's loss implementations.

When a pickle references a module named `_loss.loss`, this module will
import the equivalent implementations from `sklearn._loss.loss` and
expose them under the expected names so unpickling succeeds.
"""

try:
    # Import all public symbols from scikit-learn's loss module
    from sklearn._loss.loss import *  # type: ignore
except Exception as exc:  # pragma: no cover - runtime shim
    # Re-raise with a clearer message if scikit-learn isn't available.
    raise ImportError(
        "The compatibility shim `_loss.loss` requires scikit-learn to be "
        "installed in the runtime environment. Install scikit-learn and "
        "ensure the Python environment used by uvicorn has access to it."
    ) from exc
