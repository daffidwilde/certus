"""Tests for the `certus.parsers.struct` module."""

import json
import string
from unittest import mock

import hypothesis as hyp
import hypothesis.strategies as st
import pytest

from certus.parsers import struct

from .. import common

ST_PRIMITIVES = (
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | common.ST_STRINGS
)
ST_PRIMITIVE_LISTS = st.lists(ST_PRIMITIVES)
ST_KEYS = st.text(string.ascii_lowercase + "_")
ST_PRIMITIVE_DICTS = st.dictionaries(ST_KEYS, ST_PRIMITIVES)
ST_JSON_DATA = st.recursive(
    ST_PRIMITIVES,
    lambda children: st.lists(children) | st.dictionaries(ST_KEYS, children),
    max_leaves=50,
)


def _check_parsed_primitive_class(element, span):
    """Check a parsed primitive is the right node type for its span."""
    if len(span) > 1:
        assert element == struct.nodes.Composite(children=span)
        return

    assert element == span[0]


def _check_find_token_span(find_mock, root_obj, root_span, token_mock, kw_mock):
    """Check that the token span finder mock is called correctly."""
    calls = find_mock.call_args_list

    assert len(calls) == len(root_obj) + 1
    assert calls.pop(0) == mock.call(root_obj, token_mock, kw_mock)

    root_values = root_obj.values() if isinstance(root_obj, dict) else root_obj
    for call, value in zip(calls, root_values):
        assert call == mock.call(value, root_span, kw_mock)


@hyp.given(ST_PRIMITIVE_DICTS, common.st_token_lists(), st.data())
def test_parse_json_primitive_dict(dict_, dict_span, data):
    """
    Check the parser runs with a dictionary of primitives.

    We mock the token span finder here, telling it to spit out some
    token lists for each entry. Then we check the result is an object
    with the correct fields based on the length of the spans we provide,
    and that the span finder is called correctly.
    """
    tokens, dumps_kw = mock.Mock(), mock.Mock()
    spans = [data.draw(common.st_token_lists()) for _ in dict_]

    with mock.patch.object(struct, "_find_token_span") as find_token_span:
        find_token_span.side_effect = [dict_span, *spans]
        parsed = struct.parse_json(dict_, tokens, dumps_kw)

    assert isinstance(parsed, struct.nodes.Object)
    assert list(parsed.keys()) == list(dict_.keys())
    for element, span in zip(parsed.values(), spans):
        _check_parsed_primitive_class(element, span)

    _check_find_token_span(find_token_span, dict_, dict_span, tokens, dumps_kw)


@hyp.given(ST_PRIMITIVE_LISTS, common.st_token_lists(), st.data())
def test_parse_json_primitive_list(list_, list_span, data):
    """
    Check the parser runs with a list of primitives.

    We mock the token span finder here, telling it to spit out some
    token lists for each element. Then we check the result is an array
    with the correct elements based on the length of the spans we
    provide, and that the span finder is called correctly.
    """
    tokens, dumps_kw = mock.Mock(), mock.Mock()
    spans = [data.draw(common.st_token_lists()) for _ in list_]

    with mock.patch.object(struct, "_find_token_span") as find_token_span:
        find_token_span.side_effect = [list_span, *spans]
        parsed = struct.parse_json(list_, tokens, dumps_kw)

    assert isinstance(parsed, struct.nodes.Array)
    assert len(parsed) == len(list_)
    for element, span in zip(parsed, spans):
        _check_parsed_primitive_class(element, span)

    _check_find_token_span(find_token_span, list_, list_span, tokens, dumps_kw)


@hyp.given(ST_PRIMITIVES, common.st_token_lists())
def test_parse_json_primitive(primitive, span):
    """
    Check the parser runs with a primitive.

    We mock the token span finder here, telling it to spit out a span we
    provide. Then we check the result is of the correct class based on
    the length of the span, and that the finder is called once.
    """
    tokens, dumps_kw = mock.Mock(), mock.Mock()

    with mock.patch.object(struct, "_find_token_span", return_value=span) as find_token_span:
        parsed = struct.parse_json(primitive, tokens, dumps_kw)

    assert isinstance(parsed, (struct.nodes.Composite, struct.nodes.Token))
    _check_parsed_primitive_class(parsed, span)

    find_token_span.assert_called_once_with(primitive, tokens, dumps_kw)


def test_parse_json_raises_for_invalid_json():
    """Check the parser raises an error for anything other than JSON."""
    tokens, dumps_kw = mock.Mock(), mock.Mock()

    class NotJSON:
        pass

    with pytest.raises(ValueError, match=r"Invalid JSON data:.*NotJSON"):
        _ = struct.parse_json(NotJSON(), tokens, dumps_kw)  # pyright: ignore[reportArgumentType]


@hyp.given(ST_PRIMITIVES, common.st_token_lists())
def test_parse_json_dumps_kw_none_becomes_empty_dict(primitive, span):
    """
    Check `dumps_kw=None` is resolved as an empty dictionary.

    To make things straightforward, we only check this for primitive
    inputs. We verify this behaviour by checking how the (mocked) token
    span finder is called.
    """
    tokens = mock.Mock()

    with mock.patch.object(struct, "_find_token_span", return_value=span) as find_token_span:
        _ = struct.parse_json(primitive, tokens, dumps_kw=None)

    find_token_span.assert_called_once_with(primitive, tokens, {})


@hyp.given(ST_JSON_DATA)
def test_parse_json_recursive_node_types(json_data):
    """Check the nodes are as expected when parsing nested JSON data."""
    tokens, dumps_kw = mock.Mock(), mock.Mock()

    def _check_node_type(parsed, data):
        if isinstance(data, dict):
            assert isinstance(parsed, struct.nodes.Object)
            assert list(parsed.keys()) == list(data.keys())
            return [
                _check_node_type(pval, dval) for pval, dval in zip(parsed.values(), data.values())
            ]

        if isinstance(data, list):
            assert isinstance(parsed, struct.nodes.Array)
            return [_check_node_type(pval, dval) for pval, dval in zip(parsed, data)]

        assert isinstance(parsed, struct.nodes.Composite)

    with mock.patch.object(struct, "_find_token_span"):
        parsed = struct.parse_json(json_data, tokens, dumps_kw)

    _check_node_type(parsed, json_data)


@hyp.given(ST_PRIMITIVE_DICTS | ST_PRIMITIVE_LISTS)
def test_find_token_span_delimited_data(data):
    """Check the token span finder routes delimited data correctly."""
    tokens, dumps_kw = mock.Mock(), mock.Mock()

    with (
        mock.patch.object(struct, "_find_delimited_span") as find_delimited_span,
        mock.patch.object(struct, "_find_primitive_span") as find_primitive_span,
    ):
        span = struct._find_token_span(data, tokens, dumps_kw)

    assert span is find_delimited_span.return_value

    find_delimited_span.assert_called_once_with(tokens, type(data))
    find_primitive_span.assert_not_called()


@hyp.given(ST_PRIMITIVES)
def test_find_token_span_primitive_data(data):
    """Check the token span finder routes primitive data correctly."""
    tokens, dumps_kw = mock.Mock(), mock.Mock()

    with (
        mock.patch.object(struct, "_find_delimited_span") as find_delimited_span,
        mock.patch.object(struct, "_find_primitive_span") as find_primitive_span,
    ):
        span = struct._find_token_span(data, tokens, dumps_kw)

    assert span is find_primitive_span.return_value

    find_primitive_span.assert_called_once_with(tokens, data, dumps_kw)
    find_delimited_span.assert_not_called()


@st.composite
def st_tokenise_string(draw: st.DrawFn, string: str, start: int = 0) -> list[struct.nodes.Token]:
    """Turn a string into a list of tokens."""
    tokens, position = [], start
    while string:
        nchars = draw(st.integers(1, len(string)))
        token = struct.nodes.Token(
            value=string[:nchars], logprob=draw(common.ST_LOGPROBS), start=position
        )
        tokens.append(token)
        string = string[nchars:]
        position += nchars

    return tokens


@hyp.given(
    ST_JSON_DATA.filter(lambda d: isinstance(d, (dict, list))),
    st.just([]) | common.st_token_lists(),
    st.just([]) | common.st_token_lists(),
    st.data(),
)
def test_find_delimited_span(delimited, lpad, rpad, data):
    """
    Check the delimited span finder works for delimited data.

    To test this scenario, we build a token list from the data with
    (maybe) some extra tokens tacked-on on either side. We should get
    back the token list we built.
    """
    lpad_shift = 0 if not lpad else lpad[-1].start + len(lpad[-1].value)
    tokens = data.draw(st_tokenise_string(json.dumps(delimited), start=lpad_shift))

    tokens_shift = tokens[-1].start + len(tokens[-1].value)
    for r in rpad:
        r.start += tokens_shift

    span = struct._find_delimited_span([*lpad, *tokens, *rpad], type(delimited))

    assert isinstance(span, list)
    assert all(isinstance(tok, struct.nodes.Token) for tok in span)
    assert span == tokens
