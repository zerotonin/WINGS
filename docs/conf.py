# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add project root to path so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "W.I.N.G.S."
copyright = "2025, Yeganeh Gharabigloozare, Christoph Bleidorn, Bart Geurten"
author = "Yeganeh Gharabigloozare, Christoph Bleidorn, Bart Geurten"
release = "0.2.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google-style docstrings
    "sphinx.ext.viewcode",        # Link to source code
    "sphinx.ext.intersphinx",     # Cross-reference external docs
    "sphinx.ext.mathjax",         # LaTeX math rendering
    "sphinx_autodoc_typehints",   # Type hints in docs
    "myst_parser",                # Markdown support (.md files)
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Mock imports for optional dependencies not available in CI/docs builds
# This lets Sphinx parse gpu_abm.py without torch installed
autodoc_mock_imports = ["torch"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

# Intersphinx mapping to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Source file parsers
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "../images/wingsLogo.png"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "navigation_depth": 4,
}

# Suppress warnings about missing references to torch (optional dep)
nitpick_ignore = [
    ("py:class", "torch.Tensor"),
    ("py:class", "torch.device"),
]
