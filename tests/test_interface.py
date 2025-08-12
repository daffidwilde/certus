"""Tests for the `certus.interface` module."""

from unittest import mock

import hypothesis as hyp
import hypothesis.strategies as st

from certus import interface


def test_from_google_no_candidates():
    """Check we get nothing back without some candidates."""
    result = mock.Mock(chosen_candidates=None)
    assert interface.from_google(result) == []


@hyp.given(st.lists(st.tuples(st.text(), st.floats(max_value=0))))
def test_from_google_all_candidates(params):
    """
    Check we get a list of nodes back from a log-probability result set.

    We mock the clamp utility here, telling it to pass the
    log-probability through unchanged. Then we check the nodes match the
    result set completely, and that the clamp utility is called
    correctly.
    """
    result = mock.Mock(
        chosen_candidates=[
            mock.Mock(token=text, log_probability=logprob) for text, logprob in params
        ]
    )

    with mock.patch.object(interface.utils, "clamp") as clamp:
        clamp.side_effect = lambda val, *_, **__: val
        nodes = interface.from_google(result)

    assert isinstance(nodes, list)
    assert len(nodes) == clamp.call_count == len(params)
    assert all(isinstance(node, interface.node.TokenNode) for node in nodes)

    for can, node, call in zip(result.chosen_candidates, nodes, clamp.call_args_list):
        assert can.token == node.value
        assert can.log_probability == node.logprob
        assert call == mock.call(node.logprob, upper=0.0)
