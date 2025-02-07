# Use the official PyTorch image with GPU support
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

# Create user
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}
USER ${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=765 . .

USER root

# Set environment variables to prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary libraries for gym[box2d], pygame, and SWIG
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libjpeg-dev zlib1g-dev libfreetype6-dev libportmidi-dev pkg-config \
    swig tmux git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

USER ${MYUSER}

#disabling preallocation
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

# Uncomment below if you want jupyter 
RUN pip install jupyterlab

RUN git config --global --add safe.directory /home/${MYUSER} && \
    git config --global core.autocrlf input
