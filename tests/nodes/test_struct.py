"""Tests for the `certus.nodes.struct` module."""

import re

import hypothesis as hyp
import hypothesis.strategies as st

from certus.nodes import struct

from . import common

ST_ARRAY_BASE_ELEMENT_LISTS = st.lists(common.ST_TOKEN_NODES | common.ST_COMPOSITE_NODES)


def get_num_composites(node):
    """Get the number of explicit composite nodes in a structure."""
    if isinstance(node, common.Token):
        return 0
    
    child_num = sum(get_num_composites(child) for child in node.children)
    if isinstance(node, (struct.Array, struct.Object)):
        return child_num

    return child_num + 1


@hyp.given(ST_ARRAY_BASE_ELEMENT_LISTS)
def test_array_init_sets_children(elements):
    """Check an array sets its children to be its elements."""
    assert struct.Array(elements=elements).children == elements


@hyp.given(ST_ARRAY_BASE_ELEMENT_LISTS.filter(len), st.data())
def test_array_get_item(elements, data):
    """Check you can get an element from an array with its index."""
    idx = data.draw(st.integers(0, len(elements) - 1))
    array = struct.Array(elements=elements)

    assert array[idx] == elements[idx]


@hyp.given(ST_ARRAY_BASE_ELEMENT_LISTS)
def test_array_length(elements):
    """Check the length of an array is the length of its elements."""
    assert len(struct.Array(elements=elements)) == len(elements)


@hyp.given(ST_ARRAY_BASE_ELEMENT_LISTS)
def test_array_repr(elements):
    """Check the representation of an array is as expected."""
    array = struct.Array(elements=elements)
    repr_ = repr(array)

    assert isinstance(repr_, str)
    assert re.match(r"Array\(elements=\[.*\]\)", repr_)
    assert len(re.findall(r"children=", repr_)) == get_num_composites(array)
