FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git config --system --add safe.directory "*"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py rag_core.py repo_manager.py ./
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
