"""Helpers that write synthetic measurement files."""

import struct
from pathlib import Path

import numpy as np


def _param(name: str, value) -> bytes:
    key = name.encode("ascii").ljust(4, b"\x00")
    if isinstance(value, int):
        return key + struct.pack("<HHi", 0, 2, value)
    return key + struct.pack("<HHd", 1, 4, float(value))


def write_opus(path: Path, x: np.ndarray, y: np.ndarray, csf: float = 1.0) -> Path:
    """Write a minimal OPUS file holding one ScSm spectrum."""
    params = b"".join(
        [_param("FXV", x[0]), _param("LXV", x[-1]), _param("NPT", x.size), _param("CSF", csf)]
    )
    params += b"END\x00" + struct.pack("<HH", 0, 0)
    data = np.asarray(y, dtype="<f4").tobytes()
    n_blocks = 2
    dir_offset = 24
    param_offset = dir_offset + 12 * n_blocks
    data_offset = param_offset + len(params)
    header = struct.pack("<IdIII", 0xFEFE0A0A, 920622.0, dir_offset, n_blocks, n_blocks)
    directory = struct.pack("<BBBBii", 23, 4, 0, 0, len(params) // 4, param_offset)
    directory += struct.pack("<BBBBii", 7, 4, 0, 0, x.size, data_offset)
    path.write_bytes(header + directory + params + data)
    return path


def write_text(path: Path, x: np.ndarray, y: np.ndarray, sep: str = "\t", eol: str = "\r\n"):
    lines = [f"{a:.8f}{sep}{b:.8f}" for a, b in zip(x, y, strict=True)]
    path.write_bytes((eol.join(lines) + eol).encode())
    return path


def sweep_name(b: float, ext: str = ".txt") -> str:
    whole, frac = divmod(round(b * 1000), 1000)
    return f"Sample_4p2K_Sam1_a{whole:02d}p{frac:03d}T{ext}"
