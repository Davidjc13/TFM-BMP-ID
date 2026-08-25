# PyTorch + CUDA. En el host: driver NVIDIA y NVIDIA Container Toolkit.
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY data/checksums ./data/checksums
COPY data/README.md data/PREPROCESAMIENTO.md ./data/

# Extra [torch] omitido a propósito: la imagen ya trae PyTorch compilado con CUDA.
# Un `pip install torch` desde PyPI sustituiría el wheel GPU por el de CPU.
RUN pip install -e ".[dev]" \
    && mkdir -p data/raw data/processed results checkpoints \
    && python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"

CMD ["bash"]
