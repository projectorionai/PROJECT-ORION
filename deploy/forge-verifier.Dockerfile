# Optional verifier image; build explicitly, then select container verification.
# Generated code never installs dependencies into this image at run time.
FROM python:3.13-slim
RUN pip install --no-cache-dir mypy==1.15.0 pytest==8.3.5
WORKDIR /work
USER 65534:65534
