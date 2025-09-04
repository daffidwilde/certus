"""Tests for the `certus.nodes.core` module."""

import math
from unittest import mock

import hypothesis as hyp
import hypothesis.strategies as st
import pytest

from certus.nodes import core

from . import common

ST_EMPTY_COMPOSITE_NODES = st.builds(core.Composite, children=st.just([]))


@hyp.given(st.text(), common.ST_LOGPROBS, common.ST_STARTS)
def test_token_init(value, logprob, start):
    """Check a token is instantiated as expected."""
    token = core.Token(value, logprob, start)

    assert token.value == value
    assert token.logprob == logprob
    assert token.start == start
    assert token._confidence is None


@hyp.given(st.text(), common.ST_LOGPROBS, common.ST_STARTS)
def test_token_confidence_one_time(value, logprob, start):
    """
    Check a token calculates its confidence only once.

    We mock the clamp utility function here, telling it to pass through
    the linear probability unchanged. Then we check the confidence is
    only calculated once by accessing the property twice and checking
    this mock is called once.
    """
    with mock.patch.object(core.utils, "clamp") as clamp:
        clamp.side_effect = lambda p, _, __: p
        token = core.Token(value, logprob, start)
        c1 = token.confidence
        c2 = token.confidence

    assert c1 == c2 == math.exp(logprob)

    clamp.assert_called_once_with(c1, 0.0, 1.0)


@hyp.given(common.ST_COMPOSITE_NODES)
def test_composite_node_init(composite):
    """Check a composite is instantiated as expected."""
    children = composite.children
    assert isinstance(children, list)
    assert all(isinstance(child, (core.Token, core.Composite)) for child in children)

    assert composite._value is None
    assert composite._logprob is None
    assert composite._start is None
    assert composite._confidence is None
    assert composite._leaves is None


@hyp.given(ST_EMPTY_COMPOSITE_NODES, common.st_token_lists())
def test_composite_node_value_one_time(composite, leaves):
    """
    Check a composite calculates its value only once.

    We mock the leaf gatherer so we can ensure it is only called once,
    passing a set of leaf nodes.
    """
    with mock.patch.object(core, "gather_leaves", return_value=leaves) as gather_leaves:
        v1 = composite.value
        v2 = composite.value

    assert v1 == v2 == "".join(leaf.value for leaf in leaves)

    gather_leaves.assert_called_once_with(composite)


@hyp.given(ST_EMPTY_COMPOSITE_NODES, common.st_token_lists())
def test_composite_node_logprob_one_time(composite, leaves):
    """
    Check a composite calculates its log-probability only once.

    We mock the leaf gatherer so we can ensure it is only called once,
    passing a set of leaf nodes.
    """
    with mock.patch.object(core, "gather_leaves", return_value=leaves) as gather_leaves:
        l1 = composite.logprob
        l2 = composite.logprob

    assert l1 <= 0
    assert l1 == l2 == sum(leaf.logprob for leaf in leaves)

    gather_leaves.assert_called_once_with(composite)


@hyp.given(ST_EMPTY_COMPOSITE_NODES, common.st_token_lists())
def test_composite_node_start_one_time(composite, leaves):
    """
    Check a composite calculates its start only once.

    We mock the leaf gatherer so we can ensure it is only called once,
    passing a set of leaf nodes.
    """
    with mock.patch.object(core, "gather_leaves", return_value=leaves) as gather_leaves:
        s1 = composite.start
        s2 = composite.start

    assert s1 >= 0
    assert s1 == s2 == min(leaf.start for leaf in leaves)

    gather_leaves.assert_called_once_with(composite)


@hyp.given(ST_EMPTY_COMPOSITE_NODES, common.st_token_lists())
def test_composite_node_confidence_one_time(composite, leaves):
    """
    Check a composite calculates its confidence only once.

    We mock the leaf gatherer so we can ensure it is only called once,
    passing a set of leaf nodes. We also mock the clamp utility to pass
    the probability through unchanged.
    """
    with (
        mock.patch.object(core, "gather_leaves", return_value=leaves) as gather_leaves,
        mock.patch.object(core.utils, "clamp", side_effect=lambda p, _, __: p) as clamp,
    ):
        c1 = composite.confidence
        c2 = composite.confidence

    assert 0 <= c1 <= 1
    assert c1 == c2 == math.exp(sum(leaf.logprob for leaf in leaves) / len(leaves))

    gather_leaves.assert_called_once_with(composite)
    clamp.assert_called_once_with(c1, 0.0, 1.0)


@hyp.given(ST_EMPTY_COMPOSITE_NODES, common.st_token_lists())
def test_composite_node_leaves_one_time(composite, leaves):
    """
    Check a composite calculates its leaves only once.

    We mock the leaf gatherer so we can ensure it is only called once,
    passing a set of leaf nodes.
    """
    with mock.patch.object(core, "gather_leaves", return_value=leaves) as gather_leaves:
        l1 = composite.leaves
        l2 = composite.leaves

    assert l1 == l2 == leaves

    gather_leaves.assert_called_once_with(composite)


@hyp.given(common.ST_COMPOSITE_NODES)
def test_gather_leaves_composite_node(composite):
    """Check gathering from a composite returns a list of tokens."""
    leaves = core.gather_leaves(composite)

    def _count_leaves(node_: core.Composite | core.Token) -> int:
        if isinstance(node_, core.Token):
            return 1

        return sum(_count_leaves(child) for child in node_.children)

    assert isinstance(leaves, list)
    assert all(isinstance(leaf, core.Token) for leaf in leaves)
    assert len(leaves) == _count_leaves(composite)


@hyp.given(common.st_tokens())
def test_gather_leaves_solo_token_node(token):
    """Check gathering from a token returns itself in a list."""
    assert core.gather_leaves(token) == [token]


@hyp.given(st.builds(core.Composite, children=common.st_token_lists()))
def test_gather_leaves_composite_all_father(composite):
    """Check gathering from an all-father gives its children."""
    assert core.gather_leaves(composite) == composite.children


def test_gather_leaves_raises_for_other_node_type():
    """Check an unknown node type throws an error."""

    class NotNode:
        pass

    with pytest.raises(ValueError, match=r"Invalid node type:.*NotNode"):
        _ = core.gather_leaves(NotNode())  # type: ignore[reportArgumentType]
