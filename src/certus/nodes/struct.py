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

    def __iter__(self) -> typing.Iterator[NodeType]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(elements={self.elements!r})"


@dataclasses.dataclass(kw_only=True)
class Object(Composite):
    """
    Node representing a JSON object.

    Parameters
    ----------
    fields : dict[str, NodeType]
        Mapping from field names to child nodes.

    Attributes
    ----------
    fields : dict[str, NodeType]
        Stored mapping of field names to parsed nodes.
    """

    fields: dict[str, NodeType] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.children = list(self.fields.values())
        super().__post_init__()

    def __getitem__(self, key: str) -> NodeType:
        return self.fields[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fields={self.fields!r})"

    def keys(self) -> typing.KeysView[str]:
        """Return the object's field keys."""
        return self.fields.keys()

    def items(self) -> typing.ItemsView[str, NodeType]:
        """Return the object's (key, node) pairs."""
        return self.fields.items()

    def values(self) -> typing.ValuesView[NodeType]:
        """Return the object's field values."""
        return self.fields.values()
