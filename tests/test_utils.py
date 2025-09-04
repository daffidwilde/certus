"""Tests for the `certus.utils` module."""

import hypothesis as hyp
import hypothesis.strategies as st

from certus import utils

ST_FLOATS = st.floats(allow_nan=False)


@hyp.given(ST_FLOATS)
def test_clamp_no_limits(value):
    """Check clamping without limits does nothing."""
    assert utils.clamp(value) == value


@hyp.given(ST_FLOATS, ST_FLOATS)
def test_clamp_lower_limit(value, lower):
    """Check clamping with a lower limit sets a minimum."""
    clamped = utils.clamp(value, lower=lower)

    assert isinstance(clamped, float)
    assert clamped >= lower

    if value > lower:
        assert clamped == value
    else:
        assert clamped == lower


@hyp.given(ST_FLOATS, ST_FLOATS)
def test_clamp_upper_limit(value, upper):
    """Check clamping with an upper limits sets a maximum."""
    clamped = utils.clamp(value, upper=upper)

    assert isinstance(clamped, float)
    assert clamped <= upper

    if value < upper:
        assert clamped == value
    else:
        assert clamped == upper


@hyp.given(ST_FLOATS, st.tuples(ST_FLOATS, ST_FLOATS).map(sorted))
def test_clamp_both_limits(value, limits):
    """Check clamping with both limits sets a range."""
    lower, upper = limits
    clamped = utils.clamp(value, lower, upper)

    assert isinstance(clamped, float)
    assert lower <= clamped <= upper

    if value >= upper:
        assert clamped == upper
    elif value <= lower:
        assert clamped == lower
    else:
        assert clamped == value
