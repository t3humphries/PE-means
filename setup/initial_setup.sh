#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

#Checkout submodules
git submodule update --init --recursive

# Add our patch to FastLloyd
cd FastLloyd
git apply --reverse --check ../setup/FastLloyd.patch 2>/dev/null || \
    git apply ../setup/FastLloyd.patch
cd ..