"""Shared precision-doubling policy checks."""

import mpmath as mp
import pytest

from gravlume_research._precision import _PrecisionPolicy


def test_precision_policy_certifies_normalized_deltas() -> None:
    policy = _PrecisionPolicy(low_digits=20, high_digits=40, required_digits=10)

    with mp.workdps(50):
        evidence = policy.certify_pairs(
            (
                (mp.mpf("1"), mp.mpf("1.000000000001")),
                (mp.mpf("1e20"), mp.mpf("1.000000000001e20")),
            ),
            subject="test witness",
        )

    assert evidence.policy is policy
    assert mp.mpf("9e-13") < evidence.maximum_normalized_delta < mp.mpf("1.1e-12")


@pytest.mark.parametrize(
    "pairs",
    (
        (),
        ((mp.mpf("nan"), mp.mpf("1")),),
        ((mp.mpf("1"), mp.mpf("1.000000001")),),
    ),
)
def test_precision_policy_rejects_uncertified_evidence(
    pairs: tuple[tuple[mp.mpf, mp.mpf], ...],
) -> None:
    policy = _PrecisionPolicy(low_digits=20, high_digits=40, required_digits=10)

    with pytest.raises(AssertionError, match="test witness"):
        policy.certify_pairs(pairs, subject="test witness")
