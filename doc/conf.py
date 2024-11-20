# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
import re
from pathlib import Path

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

myst_commonmark_only = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# -- MDX conversion functions -----------------------------------------------


def convert_to_mdx(content):
    """
    Convert Sphinx-generated markdown to MDX-compatible format by:
    1. Simplifying enum representations
    2. Removing default values that contain angle brackets
    3. Updating links to match transformed headings
    """
    # Keep track of heading transformations
    heading_map = {}

    def collect_heading(match):
        """Collect original and transformed heading text for later link updates"""
        original = match.group(1)
        # Keep the heading as is, but store it for reference
        heading_map[original] = original
        return match.group(0)

    # First pass: collect all headings
    content = re.sub(r"#{1,6}\s*(.*?)\s*$", collect_heading, content, flags=re.MULTILINE)

    # Replace enum value representations
    content = re.sub(r'<(\w+)\.(\w+):\s*[\'"][^\'"]+[\'"]>', r"\1.\2", content)

    # Remove default values that contain angle brackets
    content = re.sub(r"\s*=\s*<[^>]+>", "", content)

    # Update links to match the transformed headings
    def update_link(match):
        link_text = match.group(1)
        if link_text in heading_map:
            return f'[{link_text}](#{link_text.lower().replace(".", "").replace(" ", "-")})'
        return match.group(0)

    content = re.sub(r"\[(.*?)\]\(#[^)]+\)", update_link, content)

    return content


def process_markdown_files(app, exception):
    """
    Process all markdown files in the build directory after Sphinx build is complete.
    """
    if exception is not None:  # Skip if build failed
        return

    build_dir = Path(app.outdir)

    for md_file in build_dir.glob("**/*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            converted_content = convert_to_mdx(content)

            with open(md_file, "w", encoding="utf-8") as f:
                f.write(converted_content)

            print(f"Processed {md_file}")
        except Exception as e:
            print(f"Error processing {md_file}: {str(e)}")


def setup(app):
    """
    Connect the post-build processing to Sphinx.
    """
    app.connect("build-finished", process_markdown_files)
