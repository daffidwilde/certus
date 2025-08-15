"""Commonly used utilities among the test modules."""

import hypothesis.strategies as st

from certus.nodes import Token

ST_LOGPROBS = st.floats(-10, 0)
ST_STARTS = st.integers(0, 100)
ST_TOKENS = st.builds(Token, logprob=ST_LOGPROBS, start=ST_STARTS)
ST_TOKEN_LISTS = st.lists(ST_TOKENS, min_size=1, max_size=5)
