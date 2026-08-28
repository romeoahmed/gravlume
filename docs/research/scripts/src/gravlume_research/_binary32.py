"""Binary32 primitives backed by NumPy's fixed-width scalar operations."""

import math

import numpy as np

MIN_NORMAL = float.fromhex("0x1p-126")
MIN_SUBNORMAL = float.fromhex("0x1p-149")
MAX_FINITE = float.fromhex("0x1.fffffep+127")


def round_binary32(value: float) -> float:
    """Round once to IEEE 754 binary32 and return a Python float container."""

    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float32(value))


def binary32_from_bits(bits: int) -> float:
    """Interpret an unsigned 32-bit word as an IEEE 754 binary32 value."""

    return np.asarray(bits, dtype=np.uint32).view(np.float32).item()


def next_down(value: float) -> float:
    """Return the adjacent binary32 value toward negative infinity."""

    rounded = np.float32(value)
    if math.isnan(float(rounded)):
        return math.nan
    with np.errstate(over="ignore"):
        return float(np.nextafter(rounded, np.float32(-np.inf), dtype=np.float32))


def next_up(value: float) -> float:
    """Return the adjacent binary32 value toward positive infinity."""

    rounded = np.float32(value)
    if math.isnan(float(rounded)):
        return math.nan
    with np.errstate(over="ignore"):
        return float(np.nextafter(rounded, np.float32(np.inf), dtype=np.float32))
