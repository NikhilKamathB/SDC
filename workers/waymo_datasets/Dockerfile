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
    && rm -rf /var/lib/apt/lists/*

# Install Bazelisk
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazelisk \
    && chmod +x /usr/local/bin/bazelisk

# Upgrade pip
RUN pip install --upgrade pip

# Create symlinks
RUN ln -sf /usr/local/bin/python /usr/bin/python && \
    ln -sf /usr/local/bin/python /usr/bin/python3

# Set bazelrc
RUN echo "build --python_path=/usr/local/bin/python" > /root/.bazelrc

# Clone waymo-open-dataset and build from source
WORKDIR /open-source
RUN git clone https://github.com/waymo-research/waymo-open-dataset.git && \
    cd waymo-open-dataset/src && \
    bazelisk build //waymo_open_dataset/pip_pkg_scripts:wheel
RUN mkdir -p /pkgs
RUN cd waymo-open-dataset/src && cp ./bazel-bin/waymo_open_dataset/pip_pkg_scripts/*.whl /pkgs

# Install waymo-open-dataset
WORKDIR /pkgs
RUN pip install *.whl

# Create a new user 'tyche' with the same UID as your host user
ARG USER_ID=1000
RUN useradd -m -u ${USER_ID} -s /bin/bash tyche && \
    echo "tyche ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Setup our workspace
WORKDIR /app
COPY --chown=tyche:tyche . .
RUN pip install -r requirements.txt

# Switch to the new user - This is necessary to avoid permission issues when editing files in the container from the host machine.
USER tyche