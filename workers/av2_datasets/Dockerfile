# Get python 3.10
FROM python:3.10-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"
ENV PYTHON_BIN_PATH="/usr/local/bin/python"
ENV PYTHON_LIB_PATH="/usr/local/lib/python3.10"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Create symlinks
RUN ln -sf /usr/local/bin/python /usr/bin/python && \
    ln -sf /usr/local/bin/python /usr/bin/python3

# Create a new user 'tyche' with the same UID as your host user
WORKDIR /
ARG USER_ID=1000
RUN useradd -m -u ${USER_ID} -s /bin/bash tyche && echo "tyche ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Setup Agroverse API
RUN mkdir -p /editables
RUN git clone https://github.com/argoverse/av2-api.git /editables/av2-api
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default nightly
RUN cd /editables/av2-api && cargo update
RUN pip install -e /editables/av2-api

# Setup our workspace
WORKDIR /app
COPY --chown=tyche:tyche . .
RUN pip install -r requirements.txt

# Switch to the new user - This is necessary to avoid permission issues when editing files in the container from the host machine.
USER tyche