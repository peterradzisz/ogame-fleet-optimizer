"""OGame fleet auto-optimizer.

A mixed Rust + Python project. The Rust crate ``ogame_combat`` is compiled
via maturin into a Python extension module; the Python package
``ogame_optimizer`` provides the FastAPI HTTP layer, optimizer heuristics,
and Jinja2-rendered web UI.
"""

__version__ = "0.1.0"