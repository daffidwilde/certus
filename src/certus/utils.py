"""Utility function module."""


def clamp(value: float, lower: float | None = None, upper: float | None = None) -> float:
    """
    Clamp a value according to some limit(s).

    Parameters
    ----------
    value : float
        Value to be clamped.
    lower : float, optional
        Lower limit.
    upper : float, optional
        Upper limit.
    """
    if upper is not None:
        value = min(value, upper)
    if lower is not None:
        value = max(value, lower)

    return value
