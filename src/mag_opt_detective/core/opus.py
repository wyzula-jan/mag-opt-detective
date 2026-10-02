"""Minimal reader for Bruker OPUS binary files.

Only the single-channel sample spectrum (``ScSm``) is read, which is what the
OPUS export macro writes to the ``.txt`` files used by the lab.

File layout (little endian):

* header: ``uint32 magic`` (``0xFEFE0A0A``), ``float64 version``,
  ``uint32 directory_offset``, ``uint32 max_blocks``, ``uint32 n_blocks``
* directory: ``n_blocks`` entries of 12 bytes
  ``uint8 data_type, uint8 channel_type, uint8 text_type, uint8 extended,
  int32 length (in 4-byte words), int32 offset (bytes)``
* parameter blocks: repeated ``char[4] name, uint16 type, uint16 size (in 2-byte words)``
  followed by the value, terminated by ``END``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPUS_MAGIC = 0xFEFE0A0A

# Block identifiers (data_type, channel_type)
_DATA_SPECTRUM = 7
_DATA_PARAMETERS = 23
_CHANNEL_SAMPLE_SINGLE = 4  # ScSm

_PARAM_INT = 0
_PARAM_FLOAT = 1


class OpusError(ValueError):
    """Raised when a file is not a readable OPUS file."""


@dataclass(frozen=True)
class _Block:
    data_type: int
    channel_type: int
    text_type: int
    length: int  # in 4-byte words
    offset: int  # in bytes


def is_opus_file(path: str | Path) -> bool:
    """Return True if *path* starts with the OPUS magic number."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    return len(head) == 4 and struct.unpack("<I", head)[0] == OPUS_MAGIC


def _read_directory(buf: bytes) -> list[_Block]:
    if len(buf) < 24:
        raise OpusError("file too short for an OPUS header")
    magic, _version, dir_offset, _max_blocks, n_blocks = struct.unpack_from("<IdIII", buf, 0)
    if magic != OPUS_MAGIC:
        raise OpusError("missing OPUS magic number")
    blocks = []
    for i in range(n_blocks):
        pos = dir_offset + 12 * i
        if pos + 12 > len(buf):
            raise OpusError("truncated OPUS directory")
        data_type, channel_type, text_type, _ext, length, offset = struct.unpack_from(
            "<BBBBii", buf, pos
        )
        blocks.append(_Block(data_type, channel_type, text_type, length, offset))
    return blocks


def _read_parameters(buf: bytes, block: _Block) -> dict[str, int | float | str]:
    params: dict[str, int | float | str] = {}
    pos = block.offset
    end = block.offset + 4 * block.length
    while pos + 8 <= end:
        name = buf[pos : pos + 3].decode("ascii", errors="replace")
        if name == "END":
            break
        ptype, psize = struct.unpack_from("<HH", buf, pos + 4)
        start = pos + 8
        nbytes = 2 * psize
        if ptype == _PARAM_INT:
            params[name] = struct.unpack_from("<i", buf, start)[0]
        elif ptype == _PARAM_FLOAT:
            params[name] = struct.unpack_from("<d", buf, start)[0]
        else:
            raw = buf[start : start + nbytes]
            params[name] = raw.split(b"\x00", 1)[0].decode("latin-1")
        pos = start + nbytes
    return params


def _find(blocks: list[_Block], data_type: int, channel_type: int) -> _Block:
    for block in blocks:
        if block.data_type == data_type and block.channel_type == channel_type:
            return block
    raise OpusError(f"block (type={data_type}, channel={channel_type}) not found")


def read_opus(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the ScSm spectrum of an OPUS file.

    Returns ``(x, y)`` with *x* in cm^-1 sorted ascending.
    """
    buf = Path(path).read_bytes()
    try:
        blocks = _read_directory(buf)
        data_block = _find(blocks, _DATA_SPECTRUM, _CHANNEL_SAMPLE_SINGLE)
        param_block = _find(blocks, _DATA_PARAMETERS, _CHANNEL_SAMPLE_SINGLE)
        params = _read_parameters(buf, param_block)
        npt = int(params["NPT"])
        fxv = float(params["FXV"])
        lxv = float(params["LXV"])
        csf = float(params.get("CSF", 1.0))
    except (KeyError, struct.error) as exc:
        raise OpusError(f"{Path(path).name}: cannot read ScSm spectrum ({exc})") from exc
    except OpusError as exc:
        raise OpusError(f"{Path(path).name}: {exc}") from exc

    if npt > data_block.length:
        raise OpusError(f"{Path(path).name}: NPT={npt} exceeds data block length")
    y = np.frombuffer(buf, dtype="<f4", count=npt, offset=data_block.offset).astype(np.float64)
    y *= csf
    x = np.linspace(fxv, lxv, npt)
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    return x, y
