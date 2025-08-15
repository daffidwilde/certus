"""Tests for the `certus.nodes.struct` module."""

import re
import string

import hypothesis as hyp
import hypothesis.strategies as st

from certus.nodes import Token, struct

from . import common

ST_CORE_NODES = common.st_tokens() | common.ST_COMPOSITE_NODES
ST_ARRAY_CORE_ELEMENT_LISTS = st.lists(ST_CORE_NODES)
ST_OBJECT_CORE_FIELD_DICTS = st.dictionaries(st.text(string.ascii_lowercase + "_"), ST_CORE_NODES)


def get_num_composites(node):
    """Get the number of explicit composite nodes in a structure."""
    if isinstance(node, Token):
        return 0

    child_num = sum(get_num_composites(child) for child in node.children)
    if isinstance(node, (struct.Array, struct.Object)):
        return child_num

    return child_num + 1


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
    assert len(re.findall(r"children=", repr_)) == get_num_composites(array)


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
    assert len(re.findall(r"children=", repr_)) == get_num_composites(object_)


@hyp.given(ST_OBJECT_CORE_FIELD_DICTS.filter(len))
def test_object_dict_views(fields):
    """Check you can use dictionary views on object nodes."""
    object_ = struct.Object(fields=fields)

    assert object_.keys() == fields.keys()
    assert object_.items() == fields.items()
    assert list(object_.values()) == list(fields.values())  # dict_values don't support equality
