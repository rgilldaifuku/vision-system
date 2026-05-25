#!/bin/bash
set -e

echo "Post-create setup for Vision System dev environment..."

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install dev tools
pip install pytest pytest-cov black ruff ansible

# Install Docker CLI (already in container via feature)
echo " Docker CLI available"

# Clone pr update ansible requirements
if [ ! -d "ansible/roles/external" ]; then
    mkdir -p ansible/roles/external
fi

echo "Environment setup complete"
echo ""
echo "Next Steps:"
echo "  1. Build Docker image: docker build -t vision-system:latest"
echo "  2. Test locally: docker-compose.up"
echo "  3. Provision Pi: ansible-playbook -i ansible/inventory.yml ansible/playbooks/provision-pi.yml"
echo ""