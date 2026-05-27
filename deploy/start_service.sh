#!/usr/bin/env bash
set -euo pipefail

# Enables and starts the Raspberry Pi / industrial Linux runtime service.
# Use journalctl -u vision.service -f to watch logs after startup.

sudo systemctl daemon-reload
sudo systemctl enable vision.service
sudo systemctl restart vision.service
sudo systemctl status vision.service --no-pager
