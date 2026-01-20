#!/bin/bash
# Patient Intelligence Console - Launcher Script

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "========================================="
    echo "  Patient Intelligence Console"
    echo "========================================="
    echo ""
    echo "OPENAI_API_KEY is not set."
    echo ""
    echo "Please set it by running:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "Or run the app directly with:"
    echo "  OPENAI_API_KEY='your-key' ./run.sh"
    echo ""
    exit 1
fi

echo "Starting Patient Intelligence Console..."
python -m streamlit run app.py --server.port 8501
