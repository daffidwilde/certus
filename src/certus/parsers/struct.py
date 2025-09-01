"""Module for the JSON (structured output) parser."""

import json
import re
import typing
import warnings

from certus import nodes

DELIMITERS = {dict: ("{", "}"), list: ("[", "]")}

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
    Parse some JSON data into nodes recursively into a tree.

    Parsing JSON works by identifying the span of tokens that contribute
    to each component of the data and then casting them:

    - Primitive components are cast to token nodes or composite nodes
    - Arrays (lists) are cast to array nodes
    - Objects (dictionaries) are cast to object nodes

    Parameters
    ----------
    data : JSON-like
        Data to parse.
    tokens : list of Token
        Token nodes from which to build the tree.
    dumps_kw : dict, optional
        Keyword arguments to pass to `json.dumps()` during parsing. Used
        when parsing primitive components.

    Returns
    -------
    JSON node
        Parsed token tree.
    """
    if data is not None and not isinstance(data, (str, int, float, list, dict)):
        raise ValueError(f"Invalid JSON data: {data=}, {type(data)=}")

    dumps_kw = dumps_kw or {}
    token_span = _find_token_span(data, tokens, dumps_kw)

    if isinstance(data, dict):
        fields = {k: parse_json(v, token_span, dumps_kw) for k, v in data.items()}
        return nodes.Object(fields=fields)
    if isinstance(data, list):
        elements = [parse_json(e, token_span, dumps_kw) for e in data]
        return nodes.Array(elements=elements)

    if len(token_span) == 1:
        return token_span[0]

    return nodes.Composite(children=token_span)


def _find_token_span(
    data: JSONDataType, tokens: TokenSpanType, dumps_kw: KwargsType
) -> TokenSpanType:
    """
    Find the token span of some JSON data.

    A span is the contiguous sequence of tokens required to build the
    JSON string of the provided data.

    We identify the span of a data packet by looking for its position in
    the provided token list up to some amount of whitespace in the
    scaffolding. Then the span consumes tokens until the packet is fully
    contained.

    Parameters
    ----------
    data : JSON-like
        Data for which to find the span.
    tokens : list of Token
        Tokens from which to get the span.
    dumps_kw : dict
        Keyword arguments to pass to `json.dumps()`.

    Returns
    -------
    list of Token
        Token span of the data.
    """
    pattern = _make_regex_from_json(data, dumps_kw)
    observed = "".join(t.value for t in tokens)

    search = re.search(pattern, observed, re.DOTALL)
    if search is None:
        warnings.warn(f"Unable to find span for {data=}", RuntimeWarning)
        return []

    start, end = _find_span_limits(tokens, search, pattern)

    return tokens[start:end]


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


def _find_span_limits(tokens: TokenSpanType, search: re.Match, pattern: str) -> tuple[int, int]:
    """
    Find the start and end of a token span.

    Parameters
    ----------
    tokens : list of Token
        Tokens through which to search.
    pattern : str
        Regular expression to look for when finding the span.
    search : re.Match
        Result of a search for `pattern` in the concatenated token
        string.

    Returns
    -------
    (int, int)
        Start and end (exclusive) indices for the token span.
    """
    start = max(i for i, t in enumerate(tokens) if t.start <= search.start() + tokens[0].start)

    text, end = "", start
    for token in tokens[start:]:
        if re.search(pattern, text, re.DOTALL):
            return start, end

        end += 1
        text += token.value

    return start, end
