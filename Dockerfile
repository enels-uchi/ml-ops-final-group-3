# syntax=docker/dockerfile:1
#
# Inference container for the diabetes 30-day readmission model.
# Serves src/serve_api.py (FastAPI + uvicorn) with the trained model baked into the image,
# so the container is a self-contained, reproducible unit of deployment.
#
#   docker build -t diabetes-readmission-api:1.0.0 .
#   docker run --rm -p 8000:8000 diabetes-readmission-api:1.0.0
#
# python:3.12-slim: small, and every pinned dependency (numpy / scikit-learn / xgboost)
# publishes a cp312 manylinux wheel, so nothing has to compile from source during the build.
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime XGBoost links against — without it `import xgboost` dies with
# "libgomp.so.1: cannot open shared object file". This is the Linux counterpart of the
# `brew install libomp` step the README calls out for macOS.
# curl is used by the HEALTHCHECK below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Escape hatch for networks that run TLS inspection (corporate/campus proxies re-sign PyPI
# with a private CA, which makes pip fail with CERTIFICATE_VERIFY_FAILED). Off by default so
# the normal build keeps full certificate verification:
#     docker build --build-arg PIP_TRUSTED_HOSTS=1 -t diabetes-readmission-api:1.0.0 .
ARG PIP_TRUSTED_HOSTS=0

# Dependencies get their own layer, copied before the source: requirements-serve.txt changes
# far less often than the code, so a rebuild after editing serve_api.py reuses this cached
# layer instead of re-downloading and re-installing every wheel.
COPY requirements-serve.txt ./
RUN if [ "$PIP_TRUSTED_HOSTS" = "1" ]; then \
      TRUST="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org"; \
    else TRUST=""; fi \
 && pip install --upgrade $TRUST pip \
 && pip install --no-cache-dir $TRUST -r requirements-serve.txt

# Only what the serving path touches. serve_api.py imports train_baseline for FEATURE_COLS /
# CATEGORICAL_COLS (single source of truth for preprocessing), so all of src/ comes along.
COPY src/ ./src/
COPY models/ ./models/
# Stdlib-only, a few KB: worth carrying so the container can verify itself in place with
# `docker compose exec api python deploy/smoke_test.py`.
COPY deploy/smoke_test.py ./deploy/smoke_test.py

# Non-root runtime user. The two writable paths the app needs are models/ (it persists
# feature_columns.json on startup) and monitoring/logs/ (the request log), so those are
# chowned rather than leaving the whole filesystem writable.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/monitoring/logs \
 && chown -R appuser:appuser /app
USER appuser

# Defaults; docker-compose.yml overrides these to point at a different model if needed.
ENV MODEL_PATH=models/model_baseline.pkl \
    MODEL_VERSION=baseline-1.0.0 \
    REQUEST_LOG=monitoring/logs/requests.csv

EXPOSE 8000

# start-period covers model load + pandas/xgboost import, which take a few seconds on cold start.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
