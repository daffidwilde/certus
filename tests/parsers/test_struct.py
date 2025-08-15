"""Tests for the `certus.parsers.struct` module."""

from unittest import mock

import hypothesis as hyp
import hypothesis.strategies as st
import pytest

from certus.parsers import struct

from .. import common

ST_PRIMITIVES = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text()


def _check_parsed_class(element, span):
    """Check a parsed element is the right node type for its span."""
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


@hyp.given(st.dictionaries(st.text(), ST_PRIMITIVES), common.ST_TOKEN_LISTS, st.data())
def test_parse_json_primitive_dict(dict_, dict_span, data):
    """
    Check the parser runs with a dictionary of primitives.

    We mock the token span finder here, telling it to spit out some
    token lists for each entry. Then we check the result is an object
    with the correct fields based on the length of the spans we provide,
    and that the span finder is called correctly.
    """
    tokens, dumps_kw = mock.Mock(), mock.Mock()
    spans = [data.draw(common.ST_TOKEN_LISTS) for _ in dict_]

    with mock.patch.object(struct, "_find_token_span") as find_token_span:
        find_token_span.side_effect = [dict_span, *spans]
        parsed = struct.parse_json(dict_, tokens, dumps_kw)

    assert isinstance(parsed, struct.nodes.Object)
    assert list(parsed.keys()) == list(dict_.keys())
    for element, span in zip(parsed.values(), spans):
        _check_parsed_class(element, span)

    _check_find_token_span(find_token_span, dict_, dict_span, tokens, dumps_kw)


@hyp.given(st.lists(ST_PRIMITIVES), common.ST_TOKEN_LISTS, st.data())
def test_parse_json_primitive_list(list_, list_span, data):
    """
    Check the parser runs with a list of primitives.

    We mock the token span finder here, telling it to spit out some
    token lists for each element. Then we check the result is an array
    with the correct elements based on the length of the spans we
    provide, and that the span finder is called correctly.
    """
    tokens, dumps_kw = mock.Mock(), mock.Mock()
    spans = [data.draw(common.ST_TOKEN_LISTS) for _ in list_]

    with mock.patch.object(struct, "_find_token_span") as find_token_span:
        find_token_span.side_effect = [list_span, *spans]
        parsed = struct.parse_json(list_, tokens, dumps_kw)

    assert isinstance(parsed, struct.nodes.Array)
    assert len(parsed) == len(list_)
    for element, span in zip(parsed.elements, spans):
        _check_parsed_class(element, span)

    _check_find_token_span(find_token_span, list_, list_span, tokens, dumps_kw)


@hyp.given(ST_PRIMITIVES, common.ST_TOKEN_LISTS)
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
    _check_parsed_class(parsed, span)

    find_token_span.assert_called_once_with(primitive, tokens, dumps_kw)


def test_parse_json_raises_for_invalid_json():
    """Check the parser raises an error for anything other than JSON."""
    tokens, dumps_kw = mock.Mock(), mock.Mock()

    class NotJSON:
        pass

    with pytest.raises(ValueError, match=r"Invalid JSON data:.*NotJSON"):
        _ = struct.parse_json(NotJSON(), tokens, dumps_kw)  # pyright: ignore[reportArgumentType]


@hyp.given(ST_PRIMITIVES, common.ST_TOKEN_LISTS)
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
