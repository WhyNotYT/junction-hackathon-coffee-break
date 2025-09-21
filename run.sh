#!/bin/bash

# Data Sanitizer Runner Script
# This script sets up a Python virtual environment, installs dependencies,
# cleans output directories, and runs the data sanitization pipeline

set -e  # Exit on any error

echo "=== Data Sanitizer Pipeline Starting ==="
echo "Timestamp: $(date)"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created at .venv"
else
    echo "Virtual environment already exists at .venv"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/update requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found, skipping dependency installation"
fi

# Clean output directories
echo "Cleaning output directories..."
if [ -d "output/" ]; then
    rm -rf output/*
    echo "Cleared output/ directory"
else
    echo "output/ directory does not exist, creating it..."
    mkdir -p output/
fi

if [ -d "audio_analysis/" ]; then
    rm -rf audio_analysis/*
    echo "Cleared audio_analysis/ directory"
else
    echo "audio_analysis/ directory does not exist, creating it..."
    mkdir -p audio_analysis/
fi

# Initialize log file
echo "=== Data Sanitizer Pipeline Log ===" > log.txt
echo "Started: $(date)" >> log.txt
echo "" >> log.txt

# Python scripts to run in order
scripts=(
    "csv_limiter.py"
    "firstpass.py"
    "firstpass_cleanup.py"
    "secondpass.py"
    "thirdpass_audio.py"
    "thirdpass_filter.py"
)

# Run each Python script and append output to log
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..." | tee -a log.txt
        echo "--- Output from $script ---" >> log.txt
        
        # Run script and capture both stdout and stderr
        if python "$script" >> log.txt 2>&1; then
            echo "✓ $script completed successfully" | tee -a log.txt
        else
            echo "✗ $script failed with exit code $?" | tee -a log.txt
            echo "Pipeline stopped due to error in $script"
            exit 1
        fi
        
        echo "" >> log.txt
    else
        echo "Warning: $script not found, skipping..." | tee -a log.txt
    fi
done

# Pipeline completion
echo "--- Pipeline Completed ---" >> log.txt
echo "Finished: $(date)" >> log.txt

echo "=== Data Sanitizer Pipeline Complete ==="
echo "All outputs have been logged to log.txt"
echo "Check the output/ and audio_analysis/ directories for results"

# Deactivate virtual environment
deactivate