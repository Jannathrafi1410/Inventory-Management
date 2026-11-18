# UI launcher (Streamlit)
param([string]$ApiUrl = "http://localhost:8080", [int]$Port = 8501)
$env:TBO_API_URL = $ApiUrl
.\.venv\Scripts\python.exe -m streamlit run ui/app.py --server.port $Port --server.address 0.0.0.0
