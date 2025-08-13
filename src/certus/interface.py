"""Module for the log probability interface functions."""

from google.genai import types

from . import utils
from .nodes import core


def from_google(result: types.LogprobsResult) -> list[core.Token]:
    """
    Extract token nodes from a Google GenAI log-probs result.

    Parameters
    ----------
    result : google.genai.types.LogprobsResult
        Log-probs result.

    Returns
    -------
    list of certus.nodes.Token
        Token nodes.
    """
    if result.chosen_candidates is None:
        return []

    return [
        core.Token(can.token, utils.clamp(can.log_probability, upper=0.0))
        for can in result.chosen_candidates
        if can.token is not None and can.log_probability is not None
    ]
