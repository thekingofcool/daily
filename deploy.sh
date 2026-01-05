#!/bin/bash

# Build and deploy Jekyll site to GitHub

set -e

echo "Building Jekyll site..."
bundle exec jekyll build

echo "Adding changes to git..."
git add -A

echo "Committing changes..."
git commit -m "Deploy: $(date +'%Y-%m-%d %H:%M:%S')"

echo "Pushing to GitHub..."
git push origin master

echo "Deployment complete!"