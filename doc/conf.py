# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
sys.path.insert(0, os.path.abspath("../"))

project = "diskurs"
copyright = "2024, Flurin Gishamer"
author = "Flurin Gishamer"
release = "0.0.29"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "myst_parser"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_sidebars = {"**": ["globaltoc.html"]}


# -- Options for myst_parser -------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/sphinx/intro.html#configuration

MAX_HEADING_DEPTH = 6

myst_heading_anchors = MAX_HEADING_DEPTH
