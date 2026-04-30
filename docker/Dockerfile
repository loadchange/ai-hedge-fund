FROM python:3.11-slim

WORKDIR /app

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/app

# Install uv (fast Python package manager from Astral)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# uv defaults: do not download a separate Python, use the system one
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock /app/

# Install dependencies into the system Python (no virtualenv inside the image)
RUN uv sync --frozen --no-install-project

# Copy rest of the source code
COPY . /app/

# Default command (will be overridden by Docker Compose)
CMD ["python", "src/main.py"]
