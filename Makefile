# TFM-BMP-ID — entorno local y contenedor con GPU
# Uso: make help

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON       ?= python3
VENV         := .venv
PY           := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
CONFIG       ?= configs/default.yaml
COMPOSE      ?= docker compose
IMAGE        ?= bmpid:gpu
CMD          ?=

export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

.PHONY: help install install-gpu test data features infer train-transformer eval-transformer eval-transformer-archive \
	docker-build docker-shell docker-run docker-test docker-gpu docker-smi \
	docker-data docker-features docker-infer clean

help:
	@echo "Local"
	@echo "  make install       venv + deps (sin forzar CUDA)"
	@echo "  make install-gpu   venv + PyTorch CUDA 12.4 + deps"
	@echo "  make test          pytest"
	@echo "  make data          python -m bmpid.data"
	@echo "  make features      python -m bmpid.features"
	@echo "  make infer         python -m bmpid.infer"
	@echo "  make train-transformer  Transformer + Set2set (NYIT features)"
	@echo "  make eval-transformer   EER/AUC/FAR/TTD (CHECKPOINT=...)"
	@echo ""
	@echo "Docker (NVIDIA Container Toolkit en el host / WSL2)"
	@echo "  make docker-build     construye bmpid:gpu"
	@echo "  make docker-gpu       comprueba torch.cuda y nvidia-smi"
	@echo "  make docker-smi       nvidia-smi en el contenedor"
	@echo "  make docker-shell     bash interactivo con GPU"
	@echo "  make docker-run CMD='...'  comando arbitrario"
	@echo "  make docker-test      pytest en el contenedor"
	@echo "  make docker-data      ingesta NYIT"
	@echo "  make docker-features  extrae tensores"
	@echo "  make docker-infer     stub de inferencia"
	@echo ""
	@echo "Variables: CONFIG=$(CONFIG)  CMD='python -m bmpid.features --include-imu'"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip

install: $(VENV)/bin/python
	$(PIP) install -e ".[dev]"

install-gpu: $(VENV)/bin/python
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -e ".[dev]"

test: $(VENV)/bin/python
	$(PY) -m pytest

data: $(VENV)/bin/python
	$(PY) -m bmpid.data --config $(CONFIG)

features: $(VENV)/bin/python
	$(PY) -m bmpid.features --config $(CONFIG)

infer: $(VENV)/bin/python
	$(PY) -m bmpid.infer --config $(CONFIG)

train-transformer: $(VENV)/bin/python
	$(PY) -m bmpid.train.transformer --config $(CONFIG)

eval-transformer: $(VENV)/bin/python
	$(PY) -m bmpid.eval --config $(CONFIG) --checkpoint $(or $(CHECKPOINT),checkpoints/transformer/best.pt) --split $(or $(SPLIT),eval)

eval-transformer-archive: $(VENV)/bin/python
	$(PY) -m bmpid.eval --config $(CONFIG) --checkpoint $(or $(CHECKPOINT),checkpoints/transformer/best.pt) --archive-all

docker-build:
	$(COMPOSE) build

docker-run:
	@if [ -z "$(CMD)" ]; then echo "Usa: make docker-run CMD='python -m bmpid.data'"; exit 1; fi
	$(COMPOSE) run --rm --no-deps bmpid $(CMD)

docker-shell:
	$(COMPOSE) run --rm --no-deps bmpid bash

docker-test:
	$(COMPOSE) run --rm --no-deps bmpid python -m pytest

docker-gpu:
	$(COMPOSE) run --rm --no-deps bmpid python -c "import torch; print('torch', torch.__version__); print('cuda_build', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

docker-smi:
	$(COMPOSE) run --rm --no-deps bmpid nvidia-smi

docker-data:
	$(COMPOSE) run --rm --no-deps bmpid python -m bmpid.data --config $(CONFIG)

docker-features:
	$(COMPOSE) run --rm --no-deps bmpid python -m bmpid.features --config $(CONFIG)

docker-infer:
	$(COMPOSE) run --rm --no-deps bmpid python -m bmpid.infer --config $(CONFIG)

clean:
	rm -rf $(VENV) .pytest_cache src/*.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
