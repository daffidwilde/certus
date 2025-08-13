"""Node models for structured outputs."""

import dataclasses
import typing

from .core import Composite, NodeType


@dataclasses.dataclass(kw_only=True)
class Array(Composite):
    """
    Node representing a JSON array.

    Parameters
    ----------
    elements : list of NodeType
        Ordered child nodes representing the array elements.
    """

    elements: list[NodeType] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.children = self.elements
        super().__post_init__()

    def __getitem__(self, index: int) -> NodeType:
        return self.elements[index]

    def __len__(self) -> int:
        return len(self.elements)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(elements={self.elements!r})"
