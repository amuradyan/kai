#!/usr/bin/env bash
# Install Python dependencies
# Run this after direnv loads: ./scripts/install_deps.sh

set -e

echo "Installing Python dependencies..."
echo "This will take several minutes..."

# Install dependencies
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed successfully!"
echo ""
echo "Test GPU availability with:"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'"
