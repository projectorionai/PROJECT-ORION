# O.R.I.O.N. headless cloud node — brain + private remote uplink, no GUI.
# Build (from the repo root):
#   docker build -t orion-node -f deploy/Dockerfile .
# Run:
#   docker run -d --name orion -p 8765:8765 \
#     -v orion-config:/app/config \
#     -e ORION_HEADLESS=1 orion-node
#
# Works on Oracle Cloud Ampere A1 (arm64) and x86; python:slim is multi-arch.

FROM python:3.12-slim

# Qt needs a couple of shared libs even for QtCore-only headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libegl1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Qt has no real display on a server — the offscreen platform plugin avoids
# any attempt to connect to an X server.
ENV QT_QPA_PLATFORM=offscreen \
    ORION_HEADLESS=1 \
    ORION_REMOTE_HOST=0.0.0.0 \
    ORION_REMOTE_PORT=8765 \
    PYTHONUNBUFFERED=1

COPY deploy/requirements-server.txt /app/deploy/requirements-server.txt
RUN pip install --no-cache-dir -r deploy/requirements-server.txt

COPY orion.py /app/orion.py
COPY orion_core /app/orion_core
COPY config /app/config

EXPOSE 8765

# Simple liveness probe against the uplink health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/api/health || exit 1

CMD ["python", "orion.py", "--headless"]
