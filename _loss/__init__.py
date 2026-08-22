"""
Compatibility shim package so pickles referencing a top-level
`_loss` package can resolve to scikit-learn's internal loss module.

This initializer re-exports symbols from the sibling module
`_loss.loss` (which imports from `sklearn._loss.loss`) so that
unpickling that expects attributes on the `_loss` module succeeds.
"""

try:
	# Import and re-export all public symbols from the shim module.
	from .loss import *  # noqa: F401,F403
except Exception as exc:  # pragma: no cover - runtime shim
	# If scikit-learn isn't installed, raise a clear ImportError when
	# the package is imported.
	raise ImportError(
		"The compatibility shim `_loss` requires scikit-learn to be "
		"installed in the runtime environment. Install scikit-learn and "
		"ensure the Python environment used by uvicorn has access to it."
	) from exc

__all__ = [name for name in globals() if not name.startswith("_")]
