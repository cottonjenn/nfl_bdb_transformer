#!/bin/bash
# Setup script for Football Trajectory Prediction project

echo "Setting up virtual environment..."

# Check if python3-venv is installed
if ! python3 -m venv --help &>/dev/null; then
    echo "ERROR: python3-venv is not installed."
    echo "Please install it with: sudo apt install python3.12-venv"
    echo "Or for the default Python version: sudo apt install python3-venv"
    exit 1
fi

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Virtual environment created and dependencies installed!"
echo "To activate the virtual environment, run: source .venv/bin/activate"

