#!/bin/bash

# Exit on error
set -e

echo "Building Sphinx documentation..."
cd doc
make markdown
cd ..

echo "Creating API docs directory..."
mkdir -p website/docs/api

echo "Copying Markdown files to Docusaurus..."
cp -r doc/_build/markdown/* website/docs/api

echo "Building Docusaurus site..."
cd website

echo "Cleaning Docusaurus dependencies..."
rm -rf node_modules .cache

npm install
npm run build
