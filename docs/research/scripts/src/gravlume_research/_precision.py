"""Private precision-doubling policy shared by independent research checks.

This module certifies numerical reproducibility only.  Each proof module keeps
ownership of its physical observables, discrete topology, and residuals.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import mpmath as mp


@dataclass(frozen=True, slots=True, kw_only=True)
class _PrecisionPolicy:
    """Working precisions and the minimum retained decimal-digit budget."""

    low_digits: int
    high_digits: int
    required_digits: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for value in (self.low_digits, self.high_digits, self.required_digits)
        ):
            raise TypeError("precision digits must be integers")
        if not 0 < self.required_digits <= self.low_digits < self.high_digits:
            raise ValueError("precision digits must satisfy 0 < required <= low < high")

    def certify_pairs(
        self,
        values: Iterable[tuple[mp.mpf, mp.mpf]],
        *,
        subject: str,
    ) -> _PrecisionEvidence:
        """Normalize low/high pairs, then reject incomplete or unstable evidence."""

        if not subject:
            raise ValueError("precision evidence must identify its subject")
        with mp.workdps(self.high_digits + 20):
            deltas = tuple(
                abs(low - high) / max(mp.mpf(1), abs(high)) for low, high in values
            )
            if not deltas or not all(mp.isfinite(delta) for delta in deltas):
                raise AssertionError(
                    f"{subject}: precision doubling produced empty or non-finite "
                    "evidence"
                )
            return _PrecisionEvidence(
                policy=self,
                maximum_normalized_delta=max(deltas),
                subject=subject,
            )

    def certify_delta(
        self,
        maximum_normalized_delta: mp.mpf,
        *,
        subject: str,
    ) -> _PrecisionEvidence:
        """Bind a proof-specific normalized maximum to this policy."""

        return _PrecisionEvidence(
            policy=self,
            maximum_normalized_delta=maximum_normalized_delta,
            subject=subject,
        )

    def require_metric(self, value: mp.mpf, *, subject: str) -> None:
        """Require one nonnegative finite metric to remain below the digit budget."""

        if not subject:
            raise ValueError("a precision metric must identify its subject")
        with mp.workdps(self.high_digits + 20):
            guard = mp.power(10, -self.required_digits)
            if not mp.isfinite(value) or not 0 <= value < guard:
                retained = (
                    "non-finite"
                    if not mp.isfinite(value)
                    else mp.nstr(-mp.log10(value), 12)
                    if value > 0
                    else "unbounded"
                )
                raise AssertionError(
                    f"{subject}: retained digits {retained} do not exceed the "
                    f"required {self.required_digits}"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class _PrecisionEvidence:
    """A policy-bound maximum normalized precision-doubling delta."""

    policy: _PrecisionPolicy
    maximum_normalized_delta: mp.mpf
    subject: str

    def __post_init__(self) -> None:
        self.policy.require_metric(
            self.maximum_normalized_delta,
            subject=self.subject,
        )


_SCIENTIFIC_PRECISION = _PrecisionPolicy(
    low_digits=120,
    high_digits=180,
    required_digits=80,
)
