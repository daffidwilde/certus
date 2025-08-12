"""Module for the log probability interface functions."""

from google.genai import types

from . import node, utils


def from_google(result: types.LogprobsResult) -> list[node.TokenNode]:
    """
    Extract token nodes from a Google GenAI log-probs result.

    Parameters
    ----------
    result : google.genai.types.LogprobsResult
        Log-probs result.

    Returns
    -------
    list[certus.TokenNode]
        Token nodes.
    """
    if result.chosen_candidates is None:
        return []

    return [
        node.TokenNode(can.token, utils.clamp(can.log_probability, upper=0.0))
        for can in result.chosen_candidates
        if can.token is not None and can.log_probability is not None
    ]
