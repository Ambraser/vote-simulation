FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gfortran \
        git \
        libblas-dev \
        libffi-dev \
        liblapack-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        r-base \
        r-base-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# randcorr is required by the R-backed generators.
RUN Rscript -e "install.packages('randcorr', repos='https://cloud.r-project.org')"

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY config /app/config
COPY .streamlit /app/.streamlit

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data /home/appuser

USER appuser

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    VOTE_SIM_OUTPUT_BASE_PATH=/data

EXPOSE 8501
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl --fail http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["vote-sim-ui"]
