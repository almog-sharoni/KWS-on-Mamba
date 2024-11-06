# Base image with Ubuntu, CUDA Toolkit, and cuDNN
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables for CUDA
ENV CUDA_VERSION=11.7
ENV CUDNN_VERSION=8

# Update and install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    git \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip setuptools wheel

# Install TensorFlow from PyPI
RUN pip3 install tensorflow

# Install PyTorch and its dependencies from the PyTorch index
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Jupyter
RUN pip3 install jupyter

# Set up environment variables for CUDA
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Verify CUDA installation
RUN nvcc --version

# Create a working directory
WORKDIR /workspace

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

