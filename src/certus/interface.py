"""Module for the log probability interface functions."""

import math
import google.genai.types

from .node import TokenNode


def from_google(result: google.genai.types.Candidate) -> list[TokenNode]:
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
    return [TokenNode(can.token, can.log_probability) for can in result.chosen_candidates]
