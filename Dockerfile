# Stage 1: The builder stage
FROM python:3.11-slim as builder

WORKDIR /app

ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# CHANGED: Copy requirements.txt from the 'app' subdirectory
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---

# Stage 2: The final stage
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/venv /app/venv

# CHANGED: Copy your application code from the 'app' subdirectory
COPY app/ .

ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8000

# This command should still work because we copied everything into the WORKDIR
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
