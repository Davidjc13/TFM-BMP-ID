"""IMU 36-D (réplica BehaveFormer-like). No importar desde el train del sistema propuesto.

Por timestep: acc+gyro (6), derivadas 1.ª y 2.ª (12), 3 bandas FFT locales × 6 (18).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bmpid.features.parse_nyit import parse_xy_series

IMU_DIM = 36
_FFT_WIN = 16


def imu_timestep_features(
    acc: np.ndarray,
    gyro: np.ndarray | None = None,
) -> np.ndarray:
    """`acc` [T, 4] (t, x, y, z). `gyro` misma forma o None (ceros). Salida [T, 36]."""
    acc = np.asarray(acc, dtype=np.float64)
    if acc.shape[0] == 0:
        return np.zeros((0, IMU_DIM), dtype=np.float32)
    t = acc[:, 0]
    axyz = acc[:, 1:4]
    if gyro is None or gyro.shape[0] == 0:
        gxyz = np.zeros_like(axyz)
    else:
        gyro = np.asarray(gyro, dtype=np.float64)
        gxyz = np.column_stack(
            [np.interp(t, gyro[:, 0], gyro[:, i]) for i in range(1, 4)]
        )
    six = np.concatenate([axyz, gxyz], axis=1)
    d1 = np.gradient(six, axis=0)
    d2 = np.gradient(d1, axis=0)
    fft_bands = _local_fft_bands(six, win=_FFT_WIN)
    feat = np.concatenate([six, d1, d2, fft_bands], axis=1)
    assert feat.shape[1] == IMU_DIM
    return feat.astype(np.float32)


def _local_fft_bands(six: np.ndarray, win: int) -> np.ndarray:
    t, c = six.shape
    bands = np.zeros((t, c * 3), dtype=np.float64)
    w = max(4, int(win))
    for i in range(t):
        a = max(0, i - w + 1)
        sl = six[a : i + 1]
        if sl.shape[0] < 4:
            continue
        spec = np.abs(np.fft.rfft(sl, axis=0))
        n = spec.shape[0]
        e1, e2 = max(1, n // 3), max(2, 2 * n // 3)
        low = spec[:e1].mean(axis=0)
        mid = spec[e1:e2].mean(axis=0) if e2 > e1 else low
        high = spec[e2:].mean(axis=0) if n > e2 else mid
        bands[i] = np.concatenate([low, mid, high])
    return bands


def from_csv_pair(acc_path: Path, gyro_path: Path | None) -> np.ndarray:
    acc = parse_xy_series(acc_path, n_coords=3)
    gyro = parse_xy_series(gyro_path, n_coords=3) if gyro_path and gyro_path.is_file() else None
    return imu_timestep_features(acc, gyro)


def align_to_times(imu_t: np.ndarray, imu_x: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Interpola cada canal IMU a los timestamps de la ventana de tecleo."""
    if imu_t.size == 0 or query_t.size == 0:
        return np.zeros((len(query_t), imu_x.shape[1] if imu_x.ndim == 2 else IMU_DIM), dtype=np.float32)
    cols = []
    for c in range(imu_x.shape[1]):
        cols.append(np.interp(query_t, imu_t, imu_x[:, c]))
    return np.stack(cols, axis=1).astype(np.float32)
