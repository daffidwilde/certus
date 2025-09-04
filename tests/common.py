"""Commonly used utilities among the test modules."""

import string

import hypothesis.strategies as st

from certus.nodes import Token

ST_STRINGS = st.text(string.ascii_letters + string.digits + " _-", min_size=1, max_size=5).filter(
    lambda t: not t.isspace()
)
ST_LOGPROBS = st.floats(-10, 0)
ST_STARTS = st.integers(0, 100)


@st.composite
def st_tokens(
    draw: st.DrawFn,
    value_strategy: st.SearchStrategy = ST_STRINGS,
    logprob_strategy: st.SearchStrategy = ST_LOGPROBS,
    start_strategy: st.SearchStrategy = ST_STARTS,
) -> Token:
    """Create a token for a test."""
    return draw(
        st.builds(Token, value=value_strategy, logprob=logprob_strategy, start=start_strategy)
    )


@st.composite
def st_token_lists(draw: st.DrawFn, min_size: int = 1, max_size: int = 5) -> list[Token]:
    """Create a list of tokens for a test."""
    num = draw(st.integers(min_size, max_size))
    tokens, position = [], draw(ST_STARTS)
    for _ in range(num):
        token = draw(st_tokens(start_strategy=st.just(position)))
        position += len(token.value)
        tokens.append(token)

    return tokens
