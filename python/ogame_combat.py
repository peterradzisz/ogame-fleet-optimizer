"""Top-level shim for the Rust-compiled ``ogame_combat`` extension.

The compiled cdylib lives inside the ``ogame_optimizer`` package at
``ogame_optimizer._ogame_combat`` (so ``maturin develop`` installs both the
binary and the Python package in editable mode). This shim re-exports the
binary's public surface so callers can do ``import ogame_combat`` without
the package prefix — matching the user-facing name of the Rust crate.
"""
from ogame_optimizer._ogame_combat import *  # noqa: F401,F403
from ogame_optimizer._ogame_combat import __version__  # noqa: F401

__all__ = ["__version__"]