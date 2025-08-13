"""Objects used among tests in this module."""

import hypothesis.strategies as st

from certus.nodes import Composite, Token

ST_LOGPROBS = st.floats(max_value=0)
ST_TOKEN_NODES = st.builds(Token, logprob=ST_LOGPROBS)
ST_COMPOSITE_NODES = st.recursive(
    ST_TOKEN_NODES,
    lambda children: st.builds(Composite, children=st.lists(children, min_size=1, max_size=3)),
    max_leaves=10,
).filter(lambda n: isinstance(n, Composite))
