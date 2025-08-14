"""Core node models."""

import dataclasses
import math
import typing

from certus import utils

NodeType = typing.Union["Composite", "Token"]


@dataclasses.dataclass
class Token:
    """
    Data model for a token leaf node.

    Parameters
    ----------
    value : str
        Value of the token in the output.
    logprob : float
        Log-probability of the token.
    start : int
        Position of the first token character in the response.

    Attributes
    ----------
    confidence : float
        Confidence (probability) of the token.
    """

    value: str
    logprob: float
    start: int

    def __post_init__(self):
        self._confidence: float | None = None

    @property
    def confidence(self) -> float:
        """Set or return the linear probability of the token."""
        if self._confidence is None:
            self._confidence = utils.clamp(math.exp(self.logprob), 0.0, 1.0)

        return self._confidence


@dataclasses.dataclass
class Composite:
    """
    Data model for a node made up of other nodes.

    Parameters
    ----------
    children : list of Token or Composite
        Nodes contained within this composite.

    Attributes
    ----------
    leaves : list of Token
        All leaf nodes downstream from the composite.
    value : float
        Value of the composite. Taken as the concatenation of the
        composite's leaf nodes' values separated by spaces.
    logprob : float
        Log-probability of the composite. Taken as the sum of the
        log-probability for each leaf node of the composite.
    start : int
        Position of the first character in the composite. Taken as the
        minimum of the starts for each leaf node in the composite.
    confidence : float
        Confidence of the composite. Derived as the geometric mean of
        the log-probabilities of all downstream token (leaf) nodes.
    """

    children: typing.Sequence[NodeType] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self._leaves: list[Token] | None = None
        self._value: str | None = None
        self._logprob: float | None = None
        self._start: int | None = None
        self._confidence: float | None = None

    @property
    def value(self) -> str:
        """Set or return the concatenation of the composite's values."""
        if self._value is None:
            self._value = "".join(leaf.value for leaf in self.leaves)

        return self._value

    @property
    def logprob(self) -> float:
        """Set or return the sum of the log-probs of the composite."""
        if self._logprob is None:
            self._logprob = sum(leaf.logprob for leaf in self.leaves)

        return self._logprob

    @property
    def start(self) -> int:
        """Set or return the earliest start in the composite."""
        if self._start is None:
            self._start = min(leaf.start for leaf in self.leaves)

        return self._start

    @property
    def confidence(self) -> float:
        """Set or return the confidence of the composite."""
        if self._confidence is None:
            mean_logprob = self.logprob / len(self.leaves) if self.leaves else float("-inf")
            self._confidence = utils.clamp(math.exp(mean_logprob), 0.0, 1.0)

        return self._confidence

    @property
    def leaves(self) -> list[Token]:
        """Return the leaf nodes downstream of this composite node."""
        if self._leaves is None:
            self._leaves = gather_leaves(self)

        return self._leaves


def gather_leaves(node: Token | Composite) -> list[Token]:
    """
    Get the leaf nodes downstream of a node.

    Parameters
    ----------
    node : Token or Composite
        A leaf node or one in which to delve for more.

    Returns
    -------
    list of Token
        Leaf nodes in the composite tree.
    """
    if isinstance(node, Composite):
        return [leaf for child in node.children for leaf in gather_leaves(child)]
    if isinstance(node, Token):
        return [node]

    raise ValueError(f"Invalid node type: {node}, {node.__class__}")
