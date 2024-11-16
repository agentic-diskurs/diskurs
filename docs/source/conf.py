import os
import sys

# Source code directory to the Python path
sys.path.insert(0, os.path.abspath('../../diskurs'))

# Project information
project = 'Diskurs'
copyright = '2024, Flurin Gishamer et. al'
author = 'Flurin Gishamer et. al'
release = '0.0.29'  # Should be the same as the version in pyproject.toml

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Autodoc settings
autodoc_member_order = 'bysource'  # Preserve the order of members in the source
autodoc_inherit_docstrings = True  # Inherit docstrings from base classes

autodoc_typehints = 'description'  # Include type hints in the description

# HTML output settings
html_theme = 'furo'
html_static_path = ['_static']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}
