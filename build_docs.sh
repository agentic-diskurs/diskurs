#!/bin/bash

# Exit on error
set -e

echo "Building Sphinx documentation..."
poetry run sphinx-build -b markdown docs/source docs/build/markdown

echo "Copying Markdown files to Docusaurus..."
cp -r docs/build/markdown/* docs-site/docs/

echo "Building Docusaurus site..."
cd docs-site
npm install
npm run build
