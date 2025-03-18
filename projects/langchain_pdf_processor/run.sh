#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run the FastAPI app with uvicorn in the background
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload &

# Store the process ID to be able to kill it later if needed
UVICORN_PID=$!

# Run the Streamlit frontend
streamlit run ./frontend.py

# When the Streamlit process ends, also kill the uvicorn process
kill $UVICORN_PID