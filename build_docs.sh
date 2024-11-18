#!/bin/bash

# Exit on error
set -e

echo "Building Sphinx documentation..."
poetry run sphinx-build -b markdown -W --keep-going \
    -d docs/build/doctrees docs/source docs/build/markdown

echo "Creating API docs directory..."
mkdir -p docs-site/docs/api

echo "Copying Markdown files to Docusaurus..."
cp -r docs/build/markdown/* docs-site/docs/api

echo "Building Docusaurus site..."
cd docs-site

echo "Cleaning Docusaurus dependencies..."
rm -rf node_modules .cache

npm install
npm run build
