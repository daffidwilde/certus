"""Objects used among tests in this module."""

import hypothesis.strategies as st

from certus.nodes import Composite

from ..common import ST_LOGPROBS, ST_STARTS, ST_TOKEN_LISTS, ST_TOKENS

ST_COMPOSITE_NODES = st.recursive(
    ST_TOKENS,
    lambda children: st.builds(Composite, children=st.lists(children, min_size=1, max_size=3)),
    max_leaves=10,
).filter(lambda n: isinstance(n, Composite))

__all__ = ["ST_LOGPROBS", "ST_STARTS", "ST_TOKENS", "ST_TOKEN_LISTS"]
