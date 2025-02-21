#!/bin/bash

VENV_NAME=".venv"

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_NAME
fi

if [ ! -d "$VENV_NAME" ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    source $VENV_NAME/bin/activate
    pip install -r requirements.txt
fi