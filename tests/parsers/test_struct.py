"""Tests for the `certus.parsers.struct` module."""

import itertools
import json
import re
import string
import typing
from unittest import mock

import hypothesis as hyp
import hypothesis.strategies as st
import pytest

from certus.parsers import struct

from . import common

D = typing.TypeVar("D")

ST_PRIMITIVES = (
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | common.ST_STRINGS
)
ST_PRIMITIVE_LISTS = st.lists(ST_PRIMITIVES, min_size=1)
ST_KEYS = st.text(string.ascii_lowercase + "_")
ST_PRIMITIVE_DICTS = st.dictionaries(ST_KEYS, ST_PRIMITIVES, min_size=1)
ST_JSON_DATA = st.recursive(
    ST_PRIMITIVES,
    lambda kids: st.lists(kids, min_size=1) | st.dictionaries(ST_KEYS, kids, min_size=1),
    max_leaves=50,
)


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


@st.composite
def st_span_lists(
    draw: st.DrawFn, tokens: list[struct.nodes.Token], num: int
) -> list[tuple[int, int]]:
    """Create a list of span indices for a test."""
    idx_strategy = st.integers(0, len(tokens))
    idxs = draw(st.lists(idx_strategy, min_size=num, max_size=num, unique=True).map(sorted))

    return list(itertools.pairwise(idxs))


@st.composite
def st_data_span_params(
    draw: st.DrawFn, data_strategy: st.SearchStrategy[D]
) -> tuple[D, list[struct.nodes.Token], list[tuple[int, int]]]:
    """Create a dictionary, a token list, and some spans for a test."""
    data = draw(data_strategy)
    tokens = draw(st_tokenise_string(json.dumps(data)))

    if isinstance(data, (dict, list)):
        hyp.assume(len(tokens) > len(data))

    num_items = len(data) + 2 if isinstance(data, (dict, list)) else 2
    spans = draw(st_span_lists(tokens, num_items))

    return data, tokens, spans


def _check_parsed_primitive_class(element, tokens, start, end):
    """Check a parsed primitive is the right node type for its span."""
    span = tokens[start:end]
    if len(span) > 1:
        assert element == struct.nodes.Composite(children=span)
        return

    assert element == span[0]


def _check_find_token_span(data, tokens, spans, find_mock, kw_mock):
    """Check that the token span finder mock is called correctly."""
    calls = find_mock.call_args_list

    assert len(calls) == len(data) + 1
    assert calls.pop(0) == mock.call(data, tokens, kw_mock, 0)

    data_values = data.values() if isinstance(data, dict) else data
    start = spans[0][0]
    for call, value, span in zip(calls, data_values, spans[1:]):
        assert call == mock.call(value, tokens, kw_mock, start)
        start = span[1]


@hyp.given(ST_JSON_DATA, common.st_token_lists())
def test_parse_json_main(data, tokens):
    """Check the core JSON parser runs as it should."""
    dumps_kw, node = mock.Mock(), mock.Mock()
    with mock.patch.object(struct, "_parse_json", return_value=(node, mock.Mock())) as parse_json:
        parsed = struct.parse_json(data, tokens, dumps_kw)

    assert parsed is node
    parse_json.assert_called_once_with(data, tokens, dumps_kw)


@hyp.given(ST_JSON_DATA, common.st_token_lists())
def test_parse_json_main_dumps_kw_none_becomes_empty_dict(data, tokens):
    """Check `dumps_kw=None` is resolved as an empty dictionary."""
    with mock.patch.object(
        struct, "_parse_json", return_value=(mock.Mock(), mock.Mock())
    ) as parse_json:
        _ = struct.parse_json(data, tokens, dumps_kw=None)

    parse_json.assert_called_once_with(data, tokens, {})


@hyp.given(st_data_span_params(ST_PRIMITIVE_DICTS))
def test_parse_json_primitive_dict(params):
    """
    Check the parser runs with a dictionary of primitives.

    We mock the token span finder here, telling it to spit out some
    token lists for each entry. Then we check the result is an object
    with the correct fields based on the length of the spans we provide,
    and that the span finder is called correctly.
    """
    data, tokens, spans = params
    dumps_kw = mock.Mock()

    with mock.patch.object(struct, "_find_token_span", side_effect=spans) as find_token_span:
        parsed, end = struct._parse_json(data, tokens, dumps_kw)

    assert end == spans[0][1]

    assert isinstance(parsed, struct.nodes.Object)
    assert list(parsed.keys()) == list(data.keys())
    for element, span in zip(parsed.values(), spans[1:]):
        _check_parsed_primitive_class(element, tokens, *span)

    _check_find_token_span(data, tokens, spans, find_token_span, dumps_kw)


@hyp.given(st_data_span_params(ST_PRIMITIVE_LISTS))
def test_parse_json_primitive_list(params):
    """
    Check the parser runs with a list of primitives.

    We mock the token span finder here, telling it to spit out some
    token lists for each element. Then we check the result is an array
    with the correct elements based on the length of the spans we
    provide, and that the span finder is called correctly.
    """
    data, tokens, spans = params
    dumps_kw = mock.Mock()

    with mock.patch.object(struct, "_find_token_span", side_effect=spans) as find_token_span:
        parsed, end = struct._parse_json(data, tokens, dumps_kw)

    assert end == spans[0][1]

    assert isinstance(parsed, struct.nodes.Array)
    assert len(parsed) == len(data)
    for element, span in zip(parsed, spans[1:]):
        _check_parsed_primitive_class(element, tokens, *span)

    _check_find_token_span(data, tokens, spans, find_token_span, dumps_kw)


@hyp.given(st_data_span_params(ST_PRIMITIVES))
def test_parse_json_primitive(params):
    """
    Check the parser runs with a primitive.

    We mock the token span finder here, telling it to spit out a span we
    provide. Then we check the result is of the correct class based on
    the length of the span, and that the finder is called once.
    """
    data, tokens, spans = params
    dumps_kw = mock.Mock()

    assert len(spans) == 1
    span = spans[0]

    with mock.patch.object(struct, "_find_token_span", return_value=span) as find_token_span:
        parsed, end = struct._parse_json(data, tokens, dumps_kw)

    assert end == span[1]

    assert isinstance(parsed, (struct.nodes.Composite, struct.nodes.Token))
    _check_parsed_primitive_class(parsed, tokens, *span)

    find_token_span.assert_called_once_with(data, tokens, dumps_kw, 0)


def test_parse_json_raises_for_invalid_json():
    """Check the parser raises an error for anything other than JSON."""
    tokens, dumps_kw = mock.Mock(), mock.Mock()

    class NotJSON:
        pass

    with (
        mock.patch.object(struct, "_find_token_span") as find_token_span,
        pytest.raises(ValueError, match=r"Invalid JSON data:.*NotJSON"),
    ):
        _ = struct._parse_json(NotJSON(), tokens, dumps_kw)  # pyright: ignore[reportArgumentType]

    find_token_span.assert_not_called()


@hyp.given(ST_JSON_DATA, common.st_token_lists(), ST_PRIMITIVE_DICTS, st.data())
def test_find_token_span_match(data, tokens, dumps_kw, extra):
    """Check the span-finder runs if there is a match."""
    num = len(tokens)
    offset, start, end = extra.draw(
        st.tuples(st.integers(0, num), st.integers(0, num), st.integers(0, num)).map(sorted)
    )

    with (
        mock.patch.object(struct, "_make_regex_from_json") as make_regex_from_json,
        mock.patch.object(struct, "_find_span_start", return_value=start) as find_span_start,
        mock.patch.object(struct, "_find_span_end", return_value=end) as find_span_end,
        mock.patch.object(struct.re, "search") as search,
    ):
        span = struct._find_token_span(data, tokens, dumps_kw, offset)

    assert span == (start, end)

    make_regex_from_json.assert_called_once_with(data, dumps_kw)
    search.assert_called_once_with(
        make_regex_from_json.return_value, "".join(t.value for t in tokens[offset:]), re.DOTALL
    )
    find_span_start.assert_called_once_with(tokens, search.return_value, offset)
    find_span_end.assert_called_once_with(tokens, make_regex_from_json.return_value, start)


@hyp.given(ST_JSON_DATA, common.st_token_lists(), ST_PRIMITIVE_DICTS)
def test_find_token_span_no_match(data, tokens, dumps_kw):
    """Check the span-finder raises an error if there is no match."""
    with (
        mock.patch.object(struct, "_make_regex_from_json") as make_regex_from_json,
        mock.patch.object(struct, "_find_span_start") as find_span_start,
        mock.patch.object(struct, "_find_span_end") as find_span_end,
        mock.patch.object(struct.re, "search", return_value=None) as search,
        pytest.raises(RuntimeError),
    ):
        _ = struct._find_token_span(data, tokens, dumps_kw, 0)

    make_regex_from_json.assert_called_once_with(data, dumps_kw)
    search.assert_called_once_with(
        make_regex_from_json.return_value, "".join(t.value for t in tokens), re.DOTALL
    )
    find_span_start.assert_not_called()
    find_span_end.assert_not_called()


@hyp.given(ST_PRIMITIVE_DICTS.filter(len))
def test_make_regex_from_json_dict(data):
    """Check the regex builder works for a dictionary."""
    pattern = struct._make_regex_from_json(data, {})

    assert isinstance(pattern, str)
    assert re.compile(pattern)
    assert pattern.startswith("\\{\\s*")
    assert pattern.endswith("\\s*\\}")
    assert re.fullmatch(pattern, json.dumps(data)) is not None


@hyp.given(ST_PRIMITIVE_LISTS.filter(len))
def test_make_regex_from_json_list(data):
    """Check the regex builder works for a list."""
    pattern = struct._make_regex_from_json(data, {})

    assert isinstance(pattern, str)
    assert re.compile(pattern)
    assert pattern.startswith("\\[\\s*")
    assert pattern.endswith("\\s*\\]")
    assert re.fullmatch(pattern, json.dumps(data)) is not None


@hyp.given(ST_PRIMITIVES)
def test_make_regex_from_json_primitive(data):
    """Check the regex builder works for a primitive."""
    pattern = struct._make_regex_from_json(data, {})

    assert isinstance(pattern, str)
    assert re.compile(pattern)
    assert re.fullmatch(pattern, json.dumps(data)) is not None


@hyp.given(ST_JSON_DATA)
def test_make_regex_from_json_recursive(data):
    """Check the regex builder works for nested JSON data."""
    pattern = struct._make_regex_from_json(data, {})

    assert isinstance(pattern, str)
    assert re.compile(pattern)
    assert re.fullmatch(pattern, json.dumps(data)) is not None

    opening_spans = {_.span() for _ in re.finditer(r"\\[\{\[]", pattern)}
    opening_space_spans = {_.span() for _ in re.finditer(r"\\[\{\[](?=\\s\*)", pattern)}
    assert opening_space_spans == opening_spans

    closure_spans = {_.span() for _ in re.finditer(r"\\[\}\]]", pattern)}
    closure_space_spans = {_.span() for _ in re.finditer(r"(?<=\\s\*)\\[\}\]]", pattern)}
    assert closure_space_spans == closure_spans


@st.composite
def st_dicts_with_spaces(draw: st.DrawFn) -> dict[str, struct.JSONPrimitiveType]:
    """Create a dictionary with multi-space blocks in its elements."""
    whitespace_strategy = st.text(" ", min_size=2)

    data = {}
    for key, value in draw(ST_PRIMITIVE_DICTS).items():
        key += draw(whitespace_strategy)
        if isinstance(value, str):
            value += draw(whitespace_strategy)

        data[key] = value

    return data


@hyp.given(st_dicts_with_spaces())
def test_make_regex_from_json_multispaces_only_in_strings(data):
    """
    Check that any multi-space blocks are inside string literals.

    We construct this by looking at JSON objects with string keys and
    values, where we add contiguous whitespaces.
    """
    pattern = struct._make_regex_from_json(data, {})

    multi_space_spans = {_.span() for _ in re.finditer(r"(\\ ){2,}", pattern)}
    string_literal_spans = {_.span() for _ in re.finditer(r'("(?:[^"\\]|\\.)*")', pattern)}
    for start, end in multi_space_spans:
        num_hits = sum(
            start >= string_start and end <= string_end
            for string_start, string_end in string_literal_spans
        )
        assert num_hits == 1


@hyp.given(
    ST_JSON_DATA, st.just({}) | st.fixed_dictionaries({"indent": st.sampled_from([0, 1, 2, 4])})
)
def test_make_regex_from_json_handles_indent(data, dumps_kw):
    """Check that an indent keyword is always passed to the dumper."""
    with mock.patch.object(struct.json, "dumps", side_effect=json.dumps) as dumps:
        _ = struct._make_regex_from_json(data, dumps_kw)

    dumps.assert_called_once_with(data, indent=dumps_kw.get("indent", 1))


@st.composite
def st_span_start_params(
    draw: st.DrawFn,
) -> tuple[typing.Sequence[struct.nodes.Token], int, int, int]:
    """Create tokens, a start, and offsets for a span-start test."""
    tokens = draw(common.st_token_lists(min_size=2))
    start = draw(st.integers(0, len(tokens) - 1))
    offset = draw(st.integers(0, start))

    char_start_min = sum(len(token.value) for token in tokens[offset:start])
    char_start_delta = draw(st.integers(0, len(tokens[start].value) - 1))
    char_start = char_start_min + char_start_delta

    return tokens, start, offset, char_start


@hyp.given(st_span_start_params())
def test_find_span_start_success(params):
    """
    Check the start-finder can exit successfully.

    We enforce this scenario by constructing four things:

    1. a list of tokens
    2. an expected starting index
    3. a token index offset somewhere up to the starting index
    4. a character start somewhere in the starting index token given the
       offset
    """
    tokens, start, offset, char_start = params

    search = mock.Mock()
    search.start.return_value = char_start

    idx = struct._find_span_start(tokens, search, offset)

    assert idx == start
    search.start.assert_called_once_with()


@hyp.given(common.st_token_lists(), st.data())
def test_find_span_start_failure(tokens, extra):
    """
    Check the start-finder raises an error if it does not exit.

    We pass a token list, an offset and a character start that is larger
    than the total length of the tokens.
    """
    offset = extra.draw(st.integers(0, len(tokens) - 1))

    search = mock.Mock()
    search.start.return_value = sum(len(token.value) for token in tokens) + 1

    with pytest.raises(RuntimeError, match="Unable to find start index"):
        _ = struct._find_span_start(tokens, search, offset)

    search.start.assert_called_once_with()


@st.composite
def st_span_end_params(
    draw: st.DrawFn,
) -> tuple[typing.Sequence[struct.nodes.Token], str, int, int]:
    """Create tokens, a pattern, and indices for a span-end test."""
    tokens = draw(common.st_token_lists(min_size=2))

    start = draw(st.integers(0, len(tokens) - 2))
    end = draw(st.integers(start + 1, len(tokens)))

    pattern = re.escape("".join(token.value for token in tokens[start:end]))

    return tokens, pattern, start, end


@hyp.given(st_span_end_params())
def test_find_span_end_success(params):
    """
    Check the end-finder can exit successfully.

    We ensure this scenario by constructing four things:

    1. a list of tokens
    2. a start index
    3. an expected end index
    4. a regular expression matching the concatenation of the tokens
       between these indices
    """
    tokens, pattern, start, end = params

    idx = struct._find_span_end(tokens, pattern, start)

    assert idx == end


@hyp.given(common.st_token_lists(), st.data())
def test_find_span_end_failure(tokens, extra):
    """
    Check the end-finder raises an error if it does not exit.

    We pass a token list, a starting index, and mock the regex searcher
    to always fail.
    """
    pattern = mock.Mock()
    start = extra.draw(st.integers(0, len(tokens)))

    with (
        mock.patch.object(struct.re, "search", return_value=None) as search,
        pytest.raises(RuntimeError),
    ):
        _ = struct._find_span_end(tokens, pattern, start)

    assert search.call_args_list == [
        mock.call(pattern, text, re.DOTALL)
        for text in itertools.accumulate(token.value for token in tokens[start:])
    ]
