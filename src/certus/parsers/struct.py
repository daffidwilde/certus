"""Module for the JSON (structured output) parser."""

import json
import re
import typing

from certus import nodes

JSONNodeType: typing.TypeAlias = nodes.Object | nodes.Array | nodes.Composite | nodes.Token
JSONPrimitiveType: typing.TypeAlias = None | bool | int | float | str
JSONDataType: typing.TypeAlias = (
    JSONPrimitiveType | list["JSONDataType"] | dict[str, "JSONDataType"]
)
KwargsType: typing.TypeAlias = dict[str, typing.Any]
TokenSpanType: typing.TypeAlias = typing.Sequence[nodes.Token]


def parse_json(
    data: JSONDataType, tokens: TokenSpanType, dumps_kw: KwargsType | None = None
) -> JSONNodeType:
    """
    Parse JSON recursively into a node tree.

    Parameters
    ----------
    data : JSON-like
        Data to parse.
    tokens : sequence of Token
        Token nodes.
    dumps_kw : dict, optional
        Keyword arguments for `json.dumps()`. If not provided, defaults
        to an empty dictionary.

    Raises
    ------
    ValueError
        If `data` is not valid JSON.
    RuntimeError
        If a span or element of a span cannot be found, which really
        should not happen. Ensure that `tokens` and `dumps_kw` are
        correct for your `data`.

    Returns
    -------
    JSONNodeType
        Parsed token node.
    """
    dumps_kw = dumps_kw or {}
    node, _ = _parse_json(data, tokens, dumps_kw)

    return node


def _parse_json(
    data: JSONDataType, tokens: TokenSpanType, dumps_kw: KwargsType, offset: int = 0
) -> tuple[JSONNodeType, int]:
    """
    Parse JSON into a node tree, tracking position by absolute offset.

    Parameters
    ----------
    data : JSON-like
        Data to parse.
    tokens : sequence of Token
        Token nodes.
    dumps_kw : dict
        Keyword arguments for `json.dumps()`.

    Returns
    -------
    JSONNodeType
        Parsed token node.
    int
        Index of first unused token after this subtree.
    """
    if data is not None and not isinstance(data, (str, bool, int, float, list, dict)):
        raise ValueError(f"Invalid JSON data: {data=}, {type(data)=}")

    start, end = _find_token_span(data, tokens, dumps_kw, offset)
    token_span = tokens[start:end]

    if isinstance(data, dict):
        fields = {}
        for key, value in data.items():
            node, start = _parse_json(value, tokens, dumps_kw, start)
            fields[key] = node

        return nodes.Object(fields=fields), end

    if isinstance(data, list):
        elements = []
        for item in data:
            node, start = _parse_json(item, tokens, dumps_kw, start)
            elements.append(node)

        return nodes.Array(elements=elements), end

    if len(token_span) == 1:
        return token_span[0], end

    return nodes.Composite(children=token_span), end


def _find_token_span(
    data: JSONDataType, tokens: TokenSpanType, dumps_kw: KwargsType, offset: int
) -> tuple[int, int]:
    """
    Find absolute indices for the token span of some data.

    Parameters
    ----------
    data : JSON-like
        Data to parse.
    tokens : sequence of Token
        Token nodes.
    dumps_kw : dict, optional
        Keyword arguments for `json.dumps()`.
    offset : int
        Index in `tokens` from which to start parsing.

    Returns
    -------
    tuple of (int, int)
        Start and end indices of the span.
    """
    pattern = _make_regex_from_json(data, dumps_kw)
    observed = "".join(t.value for t in tokens[offset:])

    search = re.search(pattern, observed, re.DOTALL)
    if search is None:
        raise RuntimeError(f"Unable to find span for {data=}")

    start = _find_span_start(tokens, search, offset)
    end = _find_span_end(tokens, pattern, start)

    return start, end


def _make_regex_from_json(data: JSONDataType, dumps_kw: KwargsType) -> str:
    """
    Create a regular expression from a piece of JSON data.

    The resultant pattern allows for flexible (or non-existent)
    whitespace outside string literals. We do this by enforcing
    indentation when dumping the data to a string and then iterating
    over the segments inside and outside double-quotes.

    Parameters
    ----------
    data : JSON-like
        Data to transform.
    dumps_kw : dict
        Keyword arguments to pass to `json.dumps()` when dumping `data`.

    Returns
    -------
    re.Pattern
        Regular expression of `data` with flexible whitespace.
    """
    dumps_kw = dumps_kw.copy()
    indent = dumps_kw.pop("indent", 1)
    dumped = json.dumps(data, indent=indent, **dumps_kw)

    segments = re.split(r'("(?:[^"\\]|\\.)*")', dumped)

    parts = []
    for i, segment in enumerate(segments):
        escaped = re.escape(segment)
        if i % 2 == 0:
            parts.append(re.sub(r"\s+", r"\\s*", escaped))
        else:
            parts.append(escaped)

    return re.sub(r"(\\\\s\*)+", r"\\s*", "".join(parts))


def _find_span_start(tokens: TokenSpanType, search: re.Match, offset: int) -> int:
    """
    Find the absolute start index of a span from a regex search.

    Parameters
    ----------
    tokens : sequence of Token
        The full token list.
    search : re.Match
        Match object from a regex search over the concatenated tokens.
        Used to find the local start.
    offset : int
        Index of the token where the matchable substring begins.

    Returns
    -------
    int
        Absolute start index of the token span.
    """
    char_count = 0
    char_start = search.start()
    for idx, token in enumerate(tokens[offset:], start=offset):
        char_count += len(token.value)
        if char_count > char_start:
            return idx

    raise RuntimeError(f"Unable to find start index for {search=}")


def _find_span_end(tokens: TokenSpanType, pattern: str, start: int) -> int:
    """
    Find the absolute end index of a span from a pattern.

    Parameters
    ----------
    tokens : sequence of Token
        The full token list.
    pattern : str
        The regex pattern.
    start : int
        Absolute start index of the span.

    Returns
    -------
    int
        Absolute end index (exclusive) in `tokens`.
    """
    text = ""
    for idx, token in enumerate(tokens[start:], start=start):
        text += token.value
        if re.search(pattern, text, re.DOTALL):
            return idx + 1

    raise RuntimeError(f"Unable to find end index for {pattern=}")
