"""Module for the token node class."""

import dataclasses
import math
import typing

from . import utils

NodeType = typing.Union["TokenNode", "CompositeNode"]


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
    confidence : float
        Confidence (probability) of the token.
    """

    value: str
    logprob: float

    def __post_init__(self):
        self._confidence: float | None = None

    @property
    def confidence(self) -> float:
        """Set or return the linear probability of the token."""
        if self._confidence is None:
            self._confidence = utils.clamp(math.exp(self.logprob), 0.0, 1.0)

        return self._confidence


@dataclasses.dataclass
class CompositeNode:
    """
    Data model for a node made up of other nodes.

    Parameters
    ----------
    children : list of TokenNode or CompositeNode
        Nodes contained within this composite.

    Attributes
    ----------
    confidence : float
        Confidence of the composite. Derived as the geometric mean of
        the log-probabilities of all downstream token (leaf) nodes.
    """

    children: list[NodeType]

    def __post_init__(self):
        self._value: str | None = None
        self._logprob: float | None = None
        self._confidence: float | None = None
        self._leaves: list[TokenNode] | None = None

    @property
    def value(self) -> str:
        """Set or return the concatenation of the composite's values."""
        if self._value is None:
            self._value = " ".join(leaf.value for leaf in self.leaves)

        return self._value

    @property
    def logprob(self) -> float:
        """Set or return the sum of the log-probs of the composite."""
        if self._logprob is None:
            self._logprob = sum(leaf.logprob for leaf in self.leaves)

        return self._logprob

    @property
    def confidence(self) -> float:
        """Set or return the confidence of the composite."""
        if self._confidence is None:
            mean_logprob = self.logprob / len(self.leaves)
            self._confidence = utils.clamp(math.exp(mean_logprob), 0.0, 1.0)

        return self._confidence

    @property
    def leaves(self) -> list[TokenNode]:
        """Return the leaf nodes downstream of this composite node."""
        if self._leaves is None:
            self._leaves = self._gather_leaves()

        return self._leaves

    def _gather_leaves(self) -> list[TokenNode]:
        """Get the leaf nodes downstream of this composite node."""
        leaves = []
        for child in self.children:
            if isinstance(child, TokenNode):
                leaves.append(child)
                continue

            leaves.extend(child._gather_leaves())

        return leaves
