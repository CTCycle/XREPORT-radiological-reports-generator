"""Generate the deterministic, non-clinical image/table fixture used for v3.2.0 package checks."""

from __future__ import annotations

import csv
import math
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "images"
IMAGE_SIZE = 256


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def make_image(index: int) -> bytes:
    center = IMAGE_SIZE // 2
    rows = bytearray()
    for y in range(IMAGE_SIZE):
        rows.append(0)  # PNG filter: None
        for x in range(IMAGE_SIZE):
            dx = x - center
            dy = y - center
            radius = math.sqrt(dx * dx + dy * dy)
            value = 12 + ((x * 19 + y * 23 + index * 29) % 9)
            if radius < 103:
                value += 34 + ((x * 7 + y * 11 + index * 13) % 22)
            if 38 + index % 4 <= abs(dx) <= 43 + index % 4 and abs(dy) < 78:
                value += 58
            if 62 <= radius <= 67 and (x + y + index) % 5 < 3:
                value += 42
            rows.append(min(value, 220))

    header = struct.pack(">IIBBBBB", IMAGE_SIZE, IMAGE_SIZE, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + png_chunk(b"IEND", b"")
    )


def main() -> None:
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "synthetic-dataset.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "text"))
        for index in range(1, 9):
            name = f"synthetic_{index:02d}.png"
            (IMAGE_ROOT / name).write_bytes(make_image(index))
            writer.writerow((name, f"Synthetic package validation sample {index:02d}."))


if __name__ == "__main__":
    main()
