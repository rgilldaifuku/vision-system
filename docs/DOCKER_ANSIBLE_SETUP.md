# Docker & Ansible Deployment Guide

This guide covers setting up Vision System on Raspberry Pi using Docker and Ansible.

## Quick Start: Local Development with DevContainer

If you use VS Code:

```bash
# Open the project in VS Code
code .
# Press Ctrl+Shift+P and select "Dev Containers: Reopen in Container"
```

The DevContainer will:
- Install Python 3.11 + dependencies
- Install Docker-in-Docker for local testing
- Install Ansible for provisioning

## Build Docker Image (Local or for Pi)

### Single Architecture (x86)
```bash
docker build -t vision-system:latest .
docker-compose up
```

### Multi-Architecture Build (ARM + x86)
Use `docker buildx` to build for both Pi (ARM64) and your laptop (x86):

```bash
# Set up buildx (one-time)
docker buildx create --use

# Build for multiple platforms
docker buildx build --platform linux/arm64,linux/amd64 -t yourusername/vision-system:latest .

# Load to local Docker (ARM only on Pi, x86 on laptop)
docker buildx build --platform linux/amd64 -t vision-system:latest --load .
```

## Test Locally (DevContainer or Your Machine)

```bash
# Start all services (inference + web UI)
docker-compose up

# In another terminal, check logs
docker logs vision-system

# Access web UI for review images
# http://localhost:8000
```

## Deploy to Raspberry Pi with Ansible

### Prerequisites

1. **Set up SSH key auth to Pi:**
```bash
# Generate key if needed
ssh-keygen -t ed25519 -f ~/.ssh/id_rsa_pi

# Copy public key to Pi
ssh-copy-id -i ~/.ssh/id_rsa_pi pi@192.168.1.100
```

2. **Update inventory with your Pi IP:**
Edit `ansible/inventory.yml` and set:
- `ansible_host: 192.168.1.100` (your Pi's IP)
- `pi_hostname: vision-pi-1` (descriptive name)

3. **Export your trained model:**
On your workstation:
```python
from ultralytics import YOLO
m = YOLO("models/mouse/best.pt")
m.export(format="onnx")
```
This creates `best.onnx` and avoids compiling PyTorch on the Pi.

### Run Ansible Provisioning

```bash
# Dry-run (preview what will be done)
ansible-playbook -i ansible/inventory.yml ansible/playbooks/provision-pi.yml --check

# Actual provisioning (installs packages, Docker, deploys container)
ansible-playbook -i ansible/inventory.yml ansible/playbooks/provision-pi.yml

# Re-run to update code/models (idempotent)
ansible-playbook -i ansible/inventory.yml ansible/playbooks/provision-pi.yml --tags="vision-system"
```

### What Ansible Does

1. Updates OS and installs system dependencies
2. Enables the Pi camera interface (if configured)
3. Installs Docker
4. Clones your repo (or pulls latest)
5. Copies models to Pi
6. Builds Docker image (on Pi)
7. Creates systemd service to auto-start on reboot

### Monitor Deployment on Pi

```bash
# SSH into Pi
ssh pi@192.168.1.100

# View container status
docker ps

# View logs from vision system
docker logs vision-system

# Or via systemd
sudo journalctl -u vision-system -f

# Check resource usage
docker stats
```

## File Structure

```
.
├── Dockerfile # Multi-arch (ARM/x86) container build
├── docker-compose.yml # Local testing + Pi deployment config
├── .devcontainer/ # VS Code DevContainer setup
│ ├── devcontainer.json
│ └── post-create.sh
└── ansible/
├── inventory.yml # Pi hosts (IPs, hostnames)
├── group_vars/
│ └── raspberrypis.yml # Shared Pi config (model, camera, etc.)
├── playbooks/
│ └── provision-pi.yml # Main provisioning playbook
└── roles/
├── system/ # OS setup, camera enable, directories
├── docker/ # Docker installation
└── vision-system/ # Code, models, systemd service
├── tasks/
│ └── main.yml
└── templates/
└── vision-system.service.j2
```

## Headless Inference (What Runs on Pi)

The container runs `app/headless_runner.py` by default, which:
- Loads the YOLO model
- Reads frames from `/dev/video0` (camera)
- Runs inference and logs detections
- Saves low-confidence / no-detection frames to `data/review_images/`
- Exposes logs via `docker logs` and systemd journal

You can SSH into the Pi and fetch review images:
```bash
rsync -avz pi@192.168.1.100:/home/pi/vision-system/data/review_images/ ./local-review-images/
```

## Scaling to Multiple Pis

Add more entries to `ansible/inventory.yml`:

```yaml
raspberrypis:
hosts:
pi-1:
ansible_host: 192.168.1.100
pi_hostname: vision-pi-1
pi-2:
ansible_host: 192.168.1.101
pi_hostname: vision-pi-2
pi-3:
ansible_host: 192.168.1.102
pi_hostname: vision-pi-3
```

Then provision all at once:
```bash
ansible-playbook -i ansible/inventory.yml ansible/playbooks/provision-pi.yml
```

Ansible runs tasks in parallel and reports per-host results.

## Advanced: Certificates, AWS IoT, Secrets Management

For a production multi-Pi fleet:

1. **Generate device certificates:**
```bash
ansible-vault create ansible/group_vars/certificates.yml # Encrypt certs
# Add cert/key for device auth
```

2. **AWS IoT / Lambda provisioning:**
- Create a Lambda function that mints device certs on request
- Update Ansible role to call Lambda during provisioning
- Store certs in AWS Secrets Manager or Vault

3. **OTA updates:**
- Use balena or Mender for managed device updates
- Or build a Git-based CI pipeline that builds ARM images and auto-deploys

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSH key auth fails | Ensure `~/.ssh/id_rsa_pi` exists and Pi has the public key (`ssh-copy-id`) |
| Docker build fails | Check disk space on Pi (`df -h`); may need to remove old images |
| Camera not detected | Verify `pi_camera_enabled: true` in group_vars and run `sudo raspi-config` on Pi to enable |
| Inference too slow | Switch to ONNX model export, reduce image resolution, or use a Pi 4 (8GB) |
| Models not synced | Check `rsync` in ansible role or manually copy models via `scp` |

## Next Steps

- [ ] Set up SSH key auth to your Pi
- [ ] Update `ansible/inventory.yml` with Pi IP and hostname
- [ ] Customize `ansible/group_vars/raspberrypis.yml` (camera, model, etc.)
- [ ] Export your trained model to ONNX
- [ ] Run `ansible-playbook` to provision
- [ ] Test detection: `ssh pi@IP && docker logs vision-system`
- [ ] Optional: Add more Pis to inventory and scale