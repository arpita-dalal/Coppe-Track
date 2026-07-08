#!/bin/bash

# Coppe-Track Dependency Installer
echo "Setting up dependencies for Coppe-Track..."

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please install Python 3 and try again."
    exit
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    python3 -m pip install -r requirements.txt
    echo "Installation complete!"
else
    echo "Error: requirements.txt not found in the current directory."
    exit
fi

echo "All dependencies installed. You can now run the pipeline using:"
echo "python3 Coppe-Track/src/main.py"
