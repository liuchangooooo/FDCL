#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y \
  libgtk-3-dev \
  libayatana-appindicator3-dev \
  libxdo-dev
