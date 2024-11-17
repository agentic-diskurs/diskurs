#!/bin/bash

# Exit on error
set -e

echo "Building Sphinx documentation..."
poetry run sphinx-build -b markdown docs/source docs/build/markdown

echo "Creating API docs directory..."
mkdir -p docs-site/docs/api

echo "Copying Markdown files to Docusaurus..."
cp -r docs/build/markdown/* docs-site/docs/api

echo "Building Docusaurus site..."
cd docs-site
npm install
npm run build
