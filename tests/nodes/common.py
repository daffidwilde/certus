"""Objects used among tests in this module."""

import hypothesis.strategies as st

from certus.nodes import Composite

from ..common import ST_LOGPROBS, ST_STARTS, st_token_lists, st_tokens

ST_COMPOSITE_NODES = st.recursive(
    st_tokens(),
    lambda children: st.builds(Composite, children=st.lists(children, min_size=1, max_size=3)),
    max_leaves=10,
).filter(lambda n: isinstance(n, Composite))

__all__ = ["ST_LOGPROBS", "ST_STARTS", "st_tokens", "st_token_lists"]
