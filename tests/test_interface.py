"""Tests for the `certus.interface` module."""

from unittest import mock

import hypothesis as hyp
import hypothesis.strategies as st

from certus import interface

from . import common


def test_from_google_no_candidates():
    """Check we get nothing back without some candidates."""
    result = mock.Mock(chosen_candidates=None)
    assert interface.from_google(result) == []


def _check_google_candidate_nodes(candidates, nodes, clamp):
    """Check a set of nodes derived from a Google result."""
    assert isinstance(nodes, list)
    assert len(nodes) == clamp.call_count == len(candidates)
    assert all(isinstance(node, interface.core.Token) for node in nodes)

    position = 0
    for can, node, call in zip(candidates, nodes, clamp.call_args_list):
        assert can.token == node.value
        assert can.log_probability == node.logprob
        assert position == node.start
        position += len(can.token)

        assert call == mock.call(node.logprob, upper=0.0)


@hyp.given(st.lists(st.tuples(st.text(), common.ST_LOGPROBS)))
def test_from_google_all_candidates(params):
    """
    Check we get a list of nodes back from a log-probability result set.

    We mock the clamp utility here, telling it to pass the
    log-probability through unchanged. Then we check the nodes match the
    result set, and that the clamp utility is called for each element.
    """
    result = mock.Mock(
        chosen_candidates=[
            mock.Mock(token=text, log_probability=logprob) for text, logprob in params
        ]
    )

    with mock.patch.object(interface.utils, "clamp") as clamp:
        clamp.side_effect = lambda val, *_, **__: val
        nodes = interface.from_google(result)

    _check_google_candidate_nodes(result.chosen_candidates, nodes, clamp)


@hyp.given(st.lists(st.tuples(st.none() | st.text(), st.none() | common.ST_LOGPROBS)))
def test_from_google_some_candidates(params):
    """
    Check we skip over empty candidates when building our node list.

    We mock the clamp utility here, telling it to pass the
    log-probability through unchanged. Then we check the nodes match the
    complete elements of the result set, and that the clamp utility is
    only called for them.
    """
    result = mock.Mock(
        chosen_candidates=[
            mock.Mock(token=text, log_probability=logprob) for text, logprob in params
        ]
    )

    with mock.patch.object(interface.utils, "clamp") as clamp:
        clamp.side_effect = lambda val, *_, **__: val
        nodes = interface.from_google(result)

    complete = [
        c
        for c in result.chosen_candidates
        if c.token is not None and c.log_probability is not None
    ]

    _check_google_candidate_nodes(complete, nodes, clamp)
