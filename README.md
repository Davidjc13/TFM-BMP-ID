# TFM-BMP-ID

Development of an Authentication System Based on Behavioral Biometrics Using Interaction Patterns on Mobile Devices

## Entorno

Local (CPU o GPU del host):

```bash
make install          # venv + deps
make install-gpu      # PyTorch con CUDA 12.4
make test
make data             # requiere nyit_smartphone_dataset_home_lab_cleaned/
make features
make train-transformer
make eval-transformer SPLIT=eval
```

Docker con GPU (WSL2 o Linux + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
make docker-build
make docker-gpu       # debe listar la GPU y cuda_available True
make docker-shell
make docker-test
make docker-data
make docker-features
make docker-run CMD='python -m bmpid.train.transformer --config configs/default.yaml'
```

El contenedor monta el repositorio en `/app` y reserva todas las GPUs NVIDIA (`gpus: all`). El corpus no entra en la imagen: deja `nyit_smartphone_dataset_home_lab_cleaned/` en la raíz del repo.

```bash
make help
make docker-run CMD='python -m bmpid.features --config configs/default.yaml --include-imu'
```
