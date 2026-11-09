"""Estimate IJG JPEG quality from a file's luminance quantization table.

Used to check the auxiliary ``image_quality.csv`` against the actual files
instead of trusting it.
"""

from __future__ import annotations

# IJG Annex K luminance table, natural (row-major) order.
IJG_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)  # fmt: skip

ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
)  # fmt: skip


def ijg_table(quality: int, base: tuple[int, ...] = IJG_LUMA) -> tuple[int, ...]:
    """Scaled table for an IJG quality in [1, 100] (libjpeg ``jpeg_quality_scaling``)."""
    if not 1 <= quality <= 100:
        raise ValueError(f"quality out of range: {quality}")
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return tuple(min(max((v * scale + 50) // 100, 1), 255) for v in base)


def _build_lookup() -> dict[tuple[int, ...], int]:
    lookup: dict[tuple[int, ...], int] = {}
    for q in range(100, 0, -1):  # ties resolve to the lower quality, deterministically
        natural = ijg_table(q)
        lookup[natural] = q
        lookup[tuple(natural[i] for i in ZIGZAG)] = q
    return lookup


_LOOKUP = _build_lookup()


def quality_from_table(table: list[int] | tuple[int, ...]) -> int | None:
    """Exact IJG quality for a luminance table in natural or zigzag order, else None."""
    return _LOOKUP.get(tuple(int(v) for v in table))
