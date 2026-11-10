FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS runtime
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PYTHONPATH=/workspace/src:/workspace \
    PATH=/opt/venv/bin:$PATH GIT_OPTIONAL_LOCKS=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    POLARS_MAX_THREADS=4 ALASKA2_OUTPUT_ROOT=/outputs
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.10.4
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
# Cache this dependency layer independently of source; no duplicated worktree environments or datasets.
RUN uv sync --locked --no-install-project --group analysis --group lab --group release --no-cache
FROM runtime AS lab
ENV JUPYTER_RUNTIME_DIR=/tmp/alaska2-jupyter IPYTHONDIR=/tmp/alaska2-ipython \
    MPLCONFIGDIR=/tmp/alaska2-matplotlib XDG_CACHE_HOME=/tmp/alaska2-cache
ARG ALASKA2_SOURCE_GIT_SHA
ARG ALASKA2_SOURCE_DIRTY=0
LABEL org.opencontainers.image.source="https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis"
LABEL org.opencontainers.image.revision=$ALASKA2_SOURCE_GIT_SHA
COPY . .
RUN uv sync --locked --group analysis --group lab --group release --no-cache \
    && python -m tools.write_source_revision --git-sha "$ALASKA2_SOURCE_GIT_SHA" --dirty "$ALASKA2_SOURCE_DIRTY" \
    && mkdir -p /outputs && chown 1000:1000 /outputs
USER 1000:1000
EXPOSE 8888
CMD ["jupyter", "lab", "--config=/workspace/docker/jupyter_server_config.py"]
