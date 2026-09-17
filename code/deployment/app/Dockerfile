# Stage 3 - Deployment: Web application container.

FROM python:3.11-slim

WORKDIR /app

COPY code/deployment/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/app/ /app/

# Reached by its docker-compose service name from inside the Docker network.
ENV API_URL=http://api:8000

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
