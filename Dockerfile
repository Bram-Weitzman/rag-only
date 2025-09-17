# Stage 1: The builder stage, where we install dependencies
FROM python:3.11-slim as builder

# Set the working directory
WORKDIR /app

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy and install requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---

# Stage 2: The final stage, which creates the lean production image
FROM python:3.11-slim

# Set the same working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy your application code into the container
COPY . .

# Activate the virtual environment for the final container
ENV PATH="/app/venv/bin:$PATH"

# Expose the port your application runs on (e.g., 8000 for FastAPI/Uvicorn)
EXPOSE 8000

# The command to run your application
# Replace "main:app" with the actual entrypoint of your Python app
# For example, if you run `uvicorn main:app --host 0.0.0.0 --port 8000`
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
