FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the lightweight environment file first
COPY environment.yml .

# Create the base environment
RUN conda env create -f environment.yml

# FIX 1: Set PATH so we don't need to activate the environment
ENV PATH="/opt/conda/envs/came/bin:$PATH"
ENV CONDA_DEFAULT_ENV="came"

# FIX 2: Nowoczesny PyTorch z obsługą CUDA 12.1 dla karty NVIDIA H100 (architektura sm_90)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    --default-timeout=1000 --no-cache-dir

# Install COCOAPI (używa środowiska 'came' automatycznie)
RUN mkdir -p libs && \
    cd libs && \
    git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    python setup.py build_ext install

COPY . .

RUN echo "conda activate came" >> ~/.bashrc

CMD ["python", "run.py"]