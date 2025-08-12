"""Tests for the `certus.node` module."""

import math
from unittest import mock

import hypothesis as hyp
import hypothesis.strategies as st

from certus import node

GIVEN_TOKEN_NODE_PARAMS = hyp.given(st.text(), st.floats(max_value=0))


@GIVEN_TOKEN_NODE_PARAMS
def test_token_node_init(value, logprob):
    """Check a node is instantiated as expected."""
    token = node.TokenNode(value, logprob)

    assert token.value == value
    assert token.logprob == logprob
    assert token._confidence is None


@GIVEN_TOKEN_NODE_PARAMS
def test_token_node_confidence_one_time(value, logprob):
    """
    Check a node calculates its confidence only once.

    We mock the clamp utility function here, telling it to pass through
    the linear probability unchanged. Then we check the confidence is
    only calculated once by accessing the property twice and checking
    this mock is called once.
    """
    with mock.patch.object(node.utils, "clamp") as clamp:
        clamp.side_effect = lambda p, _, __: p
        token = node.TokenNode(value, logprob)
        confidence = token.confidence
        _ = token.confidence

    assert confidence == math.exp(logprob)

    clamp.assert_called_once_with(confidence, 0.0, 1.0)
