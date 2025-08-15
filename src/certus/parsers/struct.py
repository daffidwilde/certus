"""Module for the JSON (structured output) parser."""

import json
import typing

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
    Find the token span of some JSON data given its type.

    A span is the contiguous sequence of tokens required to build the
    JSON string of the provided data.

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
    if isinstance(data, (dict, list)):
        return _find_delimited_span(tokens, type(data))

    return _find_primitive_span(tokens, data, dumps_kw)


def _find_delimited_span(tokens: TokenSpanType, kind: type[dict | list]) -> TokenSpanType:
    """
    Find the token span of some delimited JSON, i.e. arrays and objects.

    We identify the span of a delimited data packet using delimiters. We
    start by looking for the token containing the opening delimiter and
    then going until we find the token containing the closing delimiter
    at the same level.

    Parameters
    ----------
    tokens : list of Token
        Tokens from which to get the span.
    kind : type of dict or list
        Kind of delimited JSON data for which to look.

    Returns
    -------
    list of Token
        Token span of the data.
    """
    opening, closure = DELIMITERS[kind]

    start, end, depth = None, None, 0
    for i, token in enumerate(tokens):
        value = token.value
        if opening in value:
            depth += value.count(opening)
            if start is None:
                start = i

        if closure in value:
            depth -= value.count(closure)
            if depth == 0:
                end = i + 1
                break

    return tokens[start:end]


def _find_primitive_span(
    tokens: TokenSpanType, data: JSONPrimitiveType, dumps_kw: KwargsType
) -> TokenSpanType:
    """
    Find the token span of a JSON primitive.

    We identify the span of the primitive by looking for its position in
    the provided token list. Then the span consumes tokens until the
    primitive is fully contained.

    Parameters
    ----------
    tokens : list of Token
        Tokens from which to get the span.
    data : None or bool or int or float or str
        Primitive for which to look.
    dumps_kw : dict
        Keyword arguments to pass to `json.dumps()` when dumping `data`.

    Returns
    -------
    list of Token
        Token span of the data.
    """
    expected = json.dumps(data, **dumps_kw)

    observed = "".join(t.value for t in tokens)
    idx = observed.index(expected) + tokens[0].start
    start = [i for i, t in enumerate(tokens) if t.start <= idx][-1]

    text, end = "", start
    for token in tokens[start:]:
        end += 1
        if expected in text:
            break

        text += token.value

    return tokens[start:end]
