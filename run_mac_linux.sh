#!/bin/bash
echo "============================================================"
echo "  Logistic Regression — Setup and Run"
echo "============================================================"
echo ""

echo "[1/3] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/3] Installing dependencies..."
pip install -r requirements.txt --quiet

echo "[3/3] Running the ML script..."
echo ""
python logistic_regression.py

echo ""
echo "Done! Check the outputs/ folder for all plots and CSV files."
