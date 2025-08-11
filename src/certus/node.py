"""Module for the token node class."""

import dataclasses
import math


@dataclasses.dataclass
class TokenNode:
    """
    Data model for a token leaf node.

    Parameters
    ----------
    value : str
        Value of the token in the output.
    logprob : float
        Log probability of the token.
    confidence : float, optional
        Confidence (probability) of the token.
    """
    value: str
    logprob: float
    confidence: float | None = None

    def __post_init__(self):
        if self.confidence is None:
            self.confidence = self._convert(self.logprob)

    def _convert(self, logprob):
        """Convert a log probability to a probability."""
        return max(0, min(1, math.exp(logprob)))
