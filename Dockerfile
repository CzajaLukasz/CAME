FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the lightweight environment file first
COPY environment.yml .

# Create the base environment
RUN conda env create -f environment.yml

# FIX 1: Set PATH so we don't need to activate the environment
# This removes the need for "conda run" or SHELL tricks
ENV PATH="/opt/conda/envs/came/bin:$PATH"
ENV CONDA_DEFAULT_ENV="came"

# FIX 2: Install PyTorch separately with high timeout
# We use the extra-index-url to get the specific CUDA 11.0 version you originally wanted
# --default-timeout=1000 prevents the ReadTimeoutError
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    --default-timeout=1000 --no-cache-dir

# Install COCOAPI (now using the 'came' environment python automatically)
RUN mkdir -p libs && \
    cd libs && \
    git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    python setup.py build_ext install

COPY . .

RUN echo "conda activate came" >> ~/.bashrc

CMD ["python", "run.py"]