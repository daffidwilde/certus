"""Tests for the `certus.nodes.struct` module."""

import re
import string
from unittest import mock

import hypothesis as hyp
import hypothesis.strategies as st

from certus.nodes import struct

from . import common

ST_COMPOSITE_NODES = st.builds(struct.Composite, children=common.st_token_lists(min_size=2))
ST_CORE_NODES = common.st_tokens() | ST_COMPOSITE_NODES
ST_ARRAY_CORE_ELEMENT_LISTS = st.lists(ST_CORE_NODES)
ST_OBJECT_CORE_FIELD_DICTS = st.dictionaries(st.text(string.ascii_lowercase + "_"), ST_CORE_NODES)


@hyp.given(ST_CORE_NODES, st.sampled_from([int, float, str, False, True, None]))
def test_primitive_post_init(node, kind):
    """Check the primitive post-init runs as it should."""
    scaffold = mock.Mock()
    with (
        mock.patch.object(struct.Primitive, "_separate_scaffolding") as separate_scaffolding,
        mock.patch.object(struct.Primitive, "_cast_value") as cast_value,
    ):
        separate_scaffolding.return_value = (node, scaffold)
        primitive = struct.Primitive(node, kind)

    assert primitive.node == node
    assert primitive.kind == kind

    separate_scaffolding.assert_called_once_with(node)
    cast_value.assert_called_once_with(node.value, kind)

    assert primitive._scaffold == scaffold
    assert primitive.value == cast_value.return_value
    assert primitive.logprob == node.logprob
    assert primitive.start == node.start
    assert primitive.confidence == node.confidence

    if isinstance(node, struct.Token):
        assert not hasattr(primitive, "children")
        assert not hasattr(primitive, "leaves")
        return

    assert primitive.children == node.children
    assert primitive.leaves == node.leaves


@hyp.given(common.st_tokens())
def test_primitive_separate_scaffolding_token(token):
    """Check the separator does nothing for a single token."""
    with mock.patch.object(struct.Primitive, "_check_if_bad_token") as check_if_bad_token:
        node, scaffold = struct.Primitive._separate_scaffolding(token)

    assert node is token
    assert scaffold is None

    check_if_bad_token.assert_not_called()


@hyp.given(ST_COMPOSITE_NODES)
def test_primitive_separate_scaffolding_composite_no_bads(composite):
    """Check the separator works for an already clean composite node."""
    with mock.patch.object(struct.Primitive, "_check_if_bad_token") as check_if_bad_token:
        check_if_bad_token.return_value = False
        node, scaffold = struct.Primitive._separate_scaffolding(composite)

    assert node == composite
    assert scaffold is None

    assert check_if_bad_token.call_args_list == [
        mock.call(composite.leaves[0]),
        mock.call(composite.leaves[-1]),
    ]


@hyp.given(ST_COMPOSITE_NODES, st.data())
def test_primitive_separate_scaffolding_composite(composite, data):
    """Check the separator works for a dirty composite node."""
    leaves = composite.leaves
    num_leaves = len(leaves)
    is_bads = data.draw(
        st.lists(st.booleans(), min_size=num_leaves, max_size=num_leaves).filter(sum)
    )
    hyp.assume(sum(is_bads) < num_leaves and is_bads[0] is True or is_bads[-1] is True)

    with mock.patch.object(struct.Primitive, "_check_if_bad_token") as check_if_bad_token:
        check_if_bad_token.side_effect = is_bads
        node, scaffold = struct.Primitive._separate_scaffolding(composite)

    first_non_bad_front = is_bads.index(False)
    first_non_bad_back = num_leaves - is_bads[::-1].index(False) + 1
    non_bads = leaves[first_non_bad_front:first_non_bad_back]
    bads = leaves[:first_non_bad_front] + leaves[first_non_bad_back:]

    if len(non_bads) == 1:
        assert node == non_bads[0]
    else:
        assert node == struct.Composite(non_bads)

    if len(bads) == 1:
        assert scaffold == bads[0]
    else:
        assert scaffold == struct.Composite(bads)


@hyp.given(ST_ARRAY_CORE_ELEMENT_LISTS)
def test_array_init_sets_children(elements):
    """Check an array sets its children to be its elements."""
    assert struct.Array(elements=elements).children == elements


@hyp.given(ST_ARRAY_CORE_ELEMENT_LISTS.filter(len), st.data())
def test_array_get_item(elements, data):
    """Check you can get an element from an array with its index."""
    idx = data.draw(st.integers(0, len(elements) - 1))
    array = struct.Array(elements=elements)

    assert array[idx] == elements[idx]


@hyp.given(ST_ARRAY_CORE_ELEMENT_LISTS)
def test_array_iterate(elements):
    """Check you can iterate over the elements of an array naturally."""
    array = struct.Array(elements=elements)

    for i, element in enumerate(array):
        assert element == elements[i]


@hyp.given(ST_ARRAY_CORE_ELEMENT_LISTS)
def test_array_length(elements):
    """Check the length of an array is the length of its elements."""
    assert len(struct.Array(elements=elements)) == len(elements)


@hyp.given(ST_ARRAY_CORE_ELEMENT_LISTS)
def test_array_repr(elements):
    """Check the representation of an array is as expected."""
    array = struct.Array(elements=elements)
    repr_ = repr(array)

    assert isinstance(repr_, str)
    assert re.match(r"Array\(elements=\[.*\]\)", repr_)
    assert all(repr(element) in repr_ for element in elements)


@hyp.given(ST_OBJECT_CORE_FIELD_DICTS)
def test_object_sets_children(fields):
    """Check an object sets its children to be its field-values."""
    assert struct.Object(fields=fields).children == list(fields.values())


@hyp.given(ST_OBJECT_CORE_FIELD_DICTS.filter(len), st.data())
def test_object_get_item(fields, data):
    """Check you can get a value from an object with its key."""
    key = data.draw(st.sampled_from(list(fields.keys())))
    object_ = struct.Object(fields=fields)

    assert object_[key] == fields[key]


@hyp.given(ST_OBJECT_CORE_FIELD_DICTS)
def test_object_repr(fields):
    """Check the representation of an object is as expected."""
    object_ = struct.Object(fields=fields)
    repr_ = repr(object_)

    assert isinstance(repr_, str)
    assert re.match(r"Object\(fields={.*}\)", repr_)
    assert all([f"'{key}': {val!r}" in repr_ for key, val in fields.items()])


@hyp.given(ST_OBJECT_CORE_FIELD_DICTS.filter(len))
def test_object_dict_views(fields):
    """Check you can use dictionary views on object nodes."""
    object_ = struct.Object(fields=fields)

    assert object_.keys() == fields.keys()
    assert object_.items() == fields.items()
    assert list(object_.values()) == list(fields.values())  # dict_values don't support equality
