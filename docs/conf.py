"""Sphinx configuration for FuncyFlows.

Notebooks are NOT executed at build time (`nb_execution_mode = "off"`). They are checked in
without outputs so the repository stays small and the build stays fast; readers run them
themselves, which is the point. Set the mode to "auto" if you would rather ship rendered
output, but note that the notebooks train small models and the build then takes minutes.
"""
import os
import sys

# repo layout: <root>/docs/conf.py and <root>/src/FuncyFlows/, so the package is one up and
# across, not one up.
sys.path.insert(0, os.path.abspath("../src"))      # so autodoc can import FuncyFlows

project = "FuncyFlows"
author = "Liam Pinchbeck"
copyright = "2026, Liam Pinchbeck"
release = "0.2.0"

extensions = [
    "myst_nb",                      # markdown + notebooks as first-class sources
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # numpydoc-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
nb_execution_mode = "off"
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autodoc imports the package, which imports torch. Read the Docs doesn't have torch installed
# (and installing it there tends to get the build killed for memory), so the import fails, autodoc
# warns instead of erroring, and every API page renders empty while the build still goes green.
#
# So: mock the heavy dependencies only when they aren't actually there. Local builds with a real
# venv get the full thing; RTD gets mocked stand-ins, which still render signatures, docstrings and
# class hierarchies -- just thinner for anything subclassing torch.nn.Module.
try:
    import torch  # noqa: F401
except ImportError:
    autodoc_mock_imports = ["torch", "tqdm", "scipy", "matplotlib"]
    print("conf.py: torch not importable, mocking heavy deps for autodoc")

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]
html_static_path = []

try:                                                # nicer theme when it is installed
    import furo  # noqa: F401
    html_theme = "furo"
except ImportError:
    html_theme = "alabaster"

html_title = f"FuncyFlows {release}"
