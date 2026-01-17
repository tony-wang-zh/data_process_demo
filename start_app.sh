#!/usr/bin/env bash
set -e

# -----------------------------
# Config
# -----------------------------
VENV_DIR=".venv"
APP_FILE="app.py"   # change to main.py if needed

# -----------------------------
# Create venv if missing
# -----------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# -----------------------------
# Activate venv
# -----------------------------
source "$VENV_DIR/bin/activate"

# -----------------------------
# Upgrade pip (safe + recommended)
# -----------------------------
pip install --upgrade pip

# -----------------------------
# Install dependencies
# -----------------------------
if [ ! -f "requirements.txt" ]; then
  echo "ERROR: requirements.txt not found"
  exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt

# -----------------------------
# Run Streamlit app
# -----------------------------
echo "Starting Streamlit app..."
exec streamlit run "$APP_FILE"
