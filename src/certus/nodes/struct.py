"""Node models for structured outputs."""

import dataclasses
import json
import re
import typing
import warnings

from ._base import BaseNode
from .core import Composite, NodeType, Token, _make_repr

JSONNodeType: typing.TypeAlias = typing.Union["Array", "Object", "Primitive"]
PrimitiveKindType: typing.TypeAlias = type[int | float | str] | bool | None

BAD_CHARACTERS: str = '\n ,"[]{}'


@dataclasses.dataclass
class Primitive(BaseNode):
    """
    Node representing a JSON primitive.

    Accepted primitive types are: Boolean, integer, float, string, and
    null. A primitive node is effectively a pointer for another
    (composite or single-token) node.

    If the provided node is a composite of tokens, we strip out any
    non-essential tokens as "scaffolding", leaving only the token(s) for
    the primitive value itself. The stripped node is what this node
    points at, and it will have identical attributes to that node except
    its value, which we cast to its Python form.

    Parameters
    ----------
    node : Token or Composite
        Node containing the primitive data token(s).
    kind : type[int, float, str] or bool or None
        The type of the primitive, or the Python value for null and
        Boolean primitives.

    Attributes
    ----------
    value : bool or int or float or str or None
        Value of the primitive. Taken as the value of `node` cast as the
        type `kind`, or `kind` for nulls and Booleans.
    """

    node: NodeType
    kind: PrimitiveKindType

    def __repr__(self) -> str:
        return _make_repr(self)  # pyright:ignore[reportArgumentType]

    def __post_init__(self) -> None:
        self.node, self._scaffold = self._separate_scaffolding(self.node)

        self.value = self._cast_value(self.node.value, self.kind)
        self.logprob = self.node.logprob
        self.start = self.node.start
        self.confidence = self.node.confidence

        if isinstance(self.node, Composite):
            self.children = self.node.children
            self.leaves = self.node.leaves

    @staticmethod
    def _separate_scaffolding(node: NodeType) -> tuple[NodeType, NodeType | None]:
        """
        Move "bad" tokens from the perimeter of a node into scaffolding.

        Tokens are considered "bad" if they are formed only from the
        banned character list, `certus.nodes.struct.BAD_CHARACTERS`. We
        scan for bad tokens at the start and end of the node, stopping
        at each end once we find a non-bad token.

        Parameters
        ----------
        node : NodeType
            Node to scan.

        Returns
        -------
        NodeType
            Scanned node without scaffolding.
        NodeType or None
            Scaffolding token(s) or `None` if there is nothing to
            separate.
        """
        if isinstance(node, Token):
            return node, None

        good_tokens, bad_tokens = Primitive._scan_for_bad_tokens(node.leaves)

        if not good_tokens:
            warnings.warn(f"Only scaffolding tokens found in {node=}", UserWarning)
            return node, None

        scaffold = None
        if bad_tokens:
            scaffold = bad_tokens[0] if len(bad_tokens) == 1 else Composite(bad_tokens)

        node = good_tokens[0] if len(good_tokens) == 1 else Composite(good_tokens)

        return node, scaffold

    @staticmethod
    def _scan_for_bad_tokens(leaves: typing.Sequence[Token]) -> tuple[list[Token], list[Token]]:
        """Look for bad tokens on either end of a leaf sequence."""
        num_leaves = len(leaves)
        left, right = 0, num_leaves - 1

        while left < num_leaves and Primitive._check_if_bad_token(leaves[left]):
            left += 1

        while right >= 0 and Primitive._check_if_bad_token(leaves[right]):
            right -= 1

        return leaves[left:right + 1], leaves[:left] + leaves[right + 1:]


    @staticmethod
    def _check_if_bad_token(token: Token) -> bool:
        return bool(re.fullmatch(rf"[{re.escape(BAD_CHARACTERS)}]+", token.value))

    @staticmethod
    def _cast_value(value: str, kind: PrimitiveKindType) -> bool | int | float | str | None:
        """
        Attempt to cast a primitive token value to its type.

        Parameters
        ----------
        value : str
            Token value of the primitive.
        kind : type of [int | float | str] or bool or None
            Type of the primitive.

        Returns
        -------
        bool or int or float or str or None
            Cast primitive value. If casting fails, the original string
            value is returned.
        """
        if kind is None or isinstance(kind, bool):
            return kind

        try:
            return kind(json.loads(value))
        except json.JSONDecodeError:
            return kind(value)
        except ValueError:
            return value


@dataclasses.dataclass(kw_only=True)
class Array(Composite):
    """
    Node representing a JSON array.

    Parameters
    ----------
    elements : list of NodeType
        Ordered child nodes representing the array elements.
    """

    elements: list[JSONNodeType] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.children = self.elements  # pyright:ignore[reportAttributeAccessIssue]
        super().__post_init__()

    def __getitem__(self, index: int) -> JSONNodeType:
        return self.elements[index]

    def __iter__(self) -> typing.Iterator[JSONNodeType]:
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

    fields: dict[str, JSONNodeType] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.children = list(self.fields.values())  # pyright:ignore[reportAttributeAccessIssue]
        super().__post_init__()

    def __getitem__(self, key: str) -> JSONNodeType:
        return self.fields[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fields={self.fields!r})"

    def keys(self) -> typing.KeysView[str]:
        """Return the object's field keys."""
        return self.fields.keys()

    def items(self) -> typing.ItemsView[str, JSONNodeType]:
        """Return the object's (key, node) pairs."""
        return self.fields.items()

    def values(self) -> typing.ValuesView[JSONNodeType]:
        """Return the object's field values."""
        return self.fields.values()
