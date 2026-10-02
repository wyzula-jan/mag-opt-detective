"""Colour maps as colour stops, shared by the plots and the figure export (no Qt)."""

from __future__ import annotations

import numpy as np

Stop = tuple[float, tuple[int, int, int]]

# Sampled maps: "index:rrggbb" on the 0..255 scale of the 256-entry tables (magma, inferno,
# viridis and plasma from matplotlib, CC0; turbo by Google, Apache-2.0). The samples are
# chosen so that linear interpolation between them stays within 1/255 of the full table.
_SAMPLED = {
    "magma": """
        0:000004 4:020109 12:06051a 18:0c0926 27:160f3b 37:241253 42:2c115f
        48:36106b 54:400f74 62:4e117b 79:681c81 105:912b81 125:b2357b 149:d8456c
        160:e75263 165:ec5860 172:f2645c 182:f8765c 192:fc8961 206:fea36f 216:feb67c
        235:fed89a 255:fcfdbf
    """,
    "inferno": """
        0:000004 2:010106 9:040314 19:0e092b 26:180c3c 35:260c51 40:2f0a5b 44:360961
        51:420a68 64:57106e 83:751b6e 105:982766 122:b3325a 132:c13a50 146:d44842
        153:dd513a 163:e75e2e 172:ef6c23 181:f57b17 189:f8890c 194:fa9207 201:fc9f07
        204:fca50a 209:fcae12 217:fbbe23 225:f8cd37 233:f4dd4f 237:f3e55d 244:f1f179
        248:f3f68a 252:f8fb9a 255:fcffa4
    """,
    "viridis": """
        0:440154 13:481467 30:472a7a 42:443983 58:3e4c8a 81:32648e 125:218e8d
        141:1e9d89 153:22a884 162:2ab07f 174:3bbb75 188:56c667 193:60ca60 201:73d056
        215:95d840 233:c5e021 240:d8e219 246:e7e419 251:f4e61e 255:fde725
    """,
    "plasma": """
        0:0d0887 4:19068c 7:20068f 19:370499 31:4b03a1 45:6100a7 55:7100a8 65:8004a8
        68:8405a7 78:920fa3 91:a31e9a 110:ba3388 138:d5536f 158:e56a5d 173:ef7c51
        188:f79044 202:fca338 215:feb72d 224:fdc527 233:fbd324 251:f2f227 254:f0f724
        255:f0f921
    """,
    "turbo": """
        0:30123b 7:38276d 10:3b2f80 16:4040a2 22:4451bf 29:4664da 36:4776ee
        42:4685fa 48:4294ff 52:3d9efe 60:2fb2f4 66:25c0e7 74:1ad2d2 77:18d7ca
        82:18e0bd 85:1ae4b6 90:22ebaa 98:38f491 106:55fa76 112:6dfe62 117:80ff53
        122:92ff47 125:9cfe40 129:a7fc3a 137:bcf534 146:d2e935 158:ebd339 167:f7c13a
        171:fbb838 177:fea933 186:fe9029 195:f9751d 206:ed5510 210:e84b0c 218:dc3b07
        226:cc2b04 237:b21a01 248:920b01 255:7a0403
    """,
}

# The same stops as pyqtgraph's old "grey" and "bipolar" gradient presets.
_EXPLICIT: dict[str, list[Stop]] = {
    "grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "bipolar": [
        (0.0, (0, 255, 255)),
        (0.25, (0, 0, 255)),
        (0.5, (0, 0, 0)),
        (0.75, (255, 0, 0)),
        (1.0, (255, 255, 0)),
    ],
}

_NAMES = ("magma", "inferno", "viridis", "plasma", "turbo", "grey", "bipolar")


def _parse(table: str) -> list[Stop]:
    result = []
    for token in table.split():
        index, rgb = token.split(":")
        colour = (int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16))
        result.append((int(index) / 255, colour))
    return result


_STOPS: dict[str, list[Stop]] = {
    name: _parse(_SAMPLED[name]) if name in _SAMPLED else _EXPLICIT[name] for name in _NAMES
}


def names() -> tuple[str, ...]:
    """Names of the available colour maps."""
    return _NAMES


def stops(name: str) -> list[Stop]:
    """Colour stops ``(position, (r, g, b))`` of *name*, positions rising from 0 to 1."""
    try:
        return list(_STOPS[name])
    except KeyError:
        raise ValueError(f"unknown colour map {name!r}; choose from {', '.join(_NAMES)}") from None


def lut(name: str, n: int = 256) -> np.ndarray:
    """Lookup table of *n* colours (``uint8`` array of shape ``(n, 3)``)."""
    if n < 2:
        raise ValueError("a lookup table needs at least 2 entries")
    table = stops(name)
    pos = np.array([p for p, _ in table])
    colours = np.array([c for _, c in table], dtype=float)
    x = np.linspace(0.0, 1.0, n)
    channels = [np.interp(x, pos, colours[:, i]) for i in range(3)]
    return np.clip(np.rint(np.stack(channels, axis=1)), 0, 255).astype(np.uint8)
