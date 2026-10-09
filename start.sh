#!/bin/sh
# Start FastAPI backend in the background on port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 &

sleep 3 

# Start Streamlit frontend in the foreground on port 8501
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
