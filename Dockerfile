# ============================================================
# chess-sim: Local development container
#
# Base: python:3.10-slim — matches the project venv (3.10.12).
# Slim over alpine because PyTorch wheels need glibc and
# compiled C extensions (numpy, h5py, zstandard).
#
# Build:
#   docker-compose build
#
# The image installs all pip deps into a virtualenv at /opt/venv.
# Source code is NOT baked in — it is bind-mounted at runtime so
# edits on the host appear instantly inside the container.
# ============================================================

FROM python:3.10-slim AS base

# ---- System dependencies ---------------------------------------------------
# libgomp1: OpenMP runtime (PyTorch CPU threading)
# libhdf5-dev: HDF5 C library (h5py)
# libgl1: OpenGL stub (matplotlib backend, optional)
# git: needed by aim for repo metadata
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libhdf5-dev \
        libgl1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---- Non-root user ---------------------------------------------------------
RUN groupadd -r chess && useradd -r -g chess -m -s /bin/bash chess

# ---- Virtualenv setup ------------------------------------------------------
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV} && \
    chown -R chess:chess ${VIRTUAL_ENV}

# Activate venv for all subsequent RUN/CMD/ENTRYPOINT commands
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# ---- Install Python dependencies -------------------------------------------
# Copy requirements first to maximize Docker layer cache hits.
# If requirements.txt hasn't changed, pip install is skipped on rebuild.
COPY --chown=chess:chess requirements.txt /tmp/requirements.txt

# Install as non-root user into the venv
USER chess
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Working directory ------------------------------------------------------
WORKDIR /workspace

# Default: drop into an interactive bash shell with venv activated
CMD ["/bin/bash"]
