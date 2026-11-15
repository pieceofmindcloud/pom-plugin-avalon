# Builds the Avalon HIP runtime (AMD cards) in ROCm's development image: a POM
# node has no ROCm, so scripts/build-runtime.sh copies the libraries the server
# loads into build/runtime-hip/lib. Ubuntu 24.04, the POM node image's.
ARG ROCM_VERSION=7.2.4
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && (apt-get update || (sleep 30 && apt-get update)) \
    && apt-get install -y --no-install-recommends build-essential ca-certificates ccache cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*
