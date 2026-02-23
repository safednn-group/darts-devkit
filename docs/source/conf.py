# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from dataclasses import is_dataclass
project = 'DARTS-devkit'
copyright = '2026, DARTS'
author = 'DARTS'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_immaterial",
]

exclude_patterns = []
needs_id_required = False

napoleon_google_docstring = True
napoleon_use_param = True      # Use "Args:" section
napoleon_use_ivar = True       # Use "Attributes:" section

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_immaterial"
suppress_warnings = ["ref.param"]

html_js_files = [
    "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js",
    "mermaid-init.js",  # this is your local JS init file
]

def process_signature(app, what, name, obj, options, signature, return_annotation):
    # Only modify dataclass signatures
    if what == "class" and is_dataclass(obj):
        # Replace '<factory>' with '...' to make Sphinx happy
        return signature.replace("<factory>", "..."), return_annotation

def setup(app):
    app.connect("autodoc-process-signature", process_signature)