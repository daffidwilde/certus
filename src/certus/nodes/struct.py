"""Node models for structured outputs."""

import dataclasses
import json
import re
import typing

from .core import Composite, NodeType, Token, _make_repr


@dataclasses.dataclass
class Primitive:
    """
    Node representing a JSON primitive.

    Accepted primitive types are: Boolean, integer, float, string, and
    null.

    Parameters
    ----------
    node : NodeType
        Node containing the primitive data.
    kind : type[int, float, str] or bool or None
        The type of the primitive, or the Python value for null and
        Boolean primitives.

    Attributes
    ----------
    value : bool or int or float or str or None
        Value of the primitive. Taken as the value of `node` cast as the
        type `kind`.
    logprob : float
        Log-probability of `node`.
    start : int
        Position of the first character in the primitive. Taken as the
        minimum of the starts for each leaf node in the composite.
    confidence : float
        Confidence of the composite. Derived as the geometric mean of
        the log-probabilities of all downstream token (leaf) nodes.
    """

    node: NodeType
    kind: type[int | float | str] | bool | None

    BAD_CHARACTERS: typing.ClassVar[str] = '\n ,"['

    def __repr__(self) -> str:
        return _make_repr(self)

    def __post_init__(self) -> None:
        self.node, self._scaffold = self.__class__._separate_scaffolding(self.node)

    @classmethod
    def _separate_scaffolding(cls, node: NodeType) -> tuple[NodeType, NodeType | None]:
        """
        Move "bad" tokens from the perimeter of a node into scaffolding.

        Tokens are considered "bad" if they are formed only of
        characters from the banned list. We scan for bad tokens at the
        start and end of the node, stopping at each end once we find a
        non-bad token.

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

        bad_tokens, children = [], list(node.children)
        for end in (0, -1):
            while children and cls._check_if_bad_token(children[end]):
                bad_tokens.append(children.pop(end))

        scaffold = None
        if len(bad_tokens) == 1:
            scaffold = bad_tokens[0]
        if len(bad_tokens) > 1:
            scaffold = Composite(bad_tokens)

        node = children[0] if len(children) == 1 else Composite(children)

        return node, scaffold

    @classmethod
    def _check_if_bad_token(cls, token: Token) -> bool:
        """
        Check whether a token is made up of only "bad" characters.

        Parameters
        ----------
        token : Token
            Token to check.

        Returns
        -------
        bool
            Whether the token is bad or not.
        """
        return bool(re.fullmatch(rf"[{re.escape(cls.BAD_CHARACTERS)}]+", token.value))

    @property
    def value(self) -> bool | int | float | str | None:
        if self.kind in (None, False, True):
            return self.kind

        value = self.node.value
        try:
            return self.kind(json.loads(value))
        except json.JSONDecodeError:
            return self.kind(value)

    @property
    def logprob(self) -> float:
        return self.node.logprob

    @property
    def confidence(self) -> float:
        return self.node.confidence

    @property
    def start(self) -> int:
        return self.node.start

    @property
    def children(self) -> typing.Sequence[NodeType]:
        return self.node.children

    @property
    def leaves(self) -> typing.Sequence[Token]:
        return self.node.leaves


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
