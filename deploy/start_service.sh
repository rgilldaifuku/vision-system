#!/usr/bin/env bash
set -euo pipefail

sudo systemctl daemon-reload
sudo systemctl enable vision.service
sudo systemctl restart vision.service
sudo systemctl status vision.service --no-pager
