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

    tokens, position = [], 0
    for candidate in result.chosen_candidates:
        value = candidate.token
        logprob = candidate.log_probability
        if value is None or logprob is None:
            continue

        tokens.append(core.Token(value, utils.clamp(logprob, upper=0.0), position))
        position += len(value)

    return tokens
