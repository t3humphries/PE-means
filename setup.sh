#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

#Checkout submodules
git submodule update --init --recursive

# Add our patch to FastLloyd
cd FastLloyd
git apply --reverse --check ../environments/FastLloyd.patch 2>/dev/null || \
    git apply ../environments/FastLloyd.patch
cd ..