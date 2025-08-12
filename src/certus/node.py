"""Module for the token node class."""

import dataclasses
import math

from . import utils


@dataclasses.dataclass
class TokenNode:
    """
    Data model for a token leaf node.

    Parameters
    ----------
    value : str
        Value of the token in the output.
    logprob : float
        Log-probability of the token.

    Attributes
    ----------
    confidence : float, optional
        Confidence (probability) of the token.
    """

    value: str
    logprob: float

    def __post_init__(self):
        self._confidence = None

    @property
    def confidence(self) -> float:
        """Set or return the linear probability of the token."""
        if self._confidence is None:
            self._confidence = utils.clamp(math.exp(self.logprob), 0.0, 1.0)

        return self._confidence
