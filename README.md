# Certus: understanding LLM certainty

Certus allows you to estimate _confidence_ in a LLM response, both as a whole and in each part. It does this by parsing
the _log-probabilities_ from your response into a tree of _nodes_.

We build this tree from an ordered collection of `certus.nodes.core.Token` instances and gathering them up recursively
into a tree matching the structure of the response. Each `Token` is considered a leaf node in the tree, and higher-up
nodes in the tree are of other types.

## Installation

The most convenient way to install Certus is to do so from PyPI:

```bash
python -m pip install certus
```

### Developers

If you are planning to do some development work on Certus, please install the package from source and use `uv`:

```bash
git clone https://github.com/daffidwilde/certus
cd certus
uv sync --dev
```

## Usage

### Extracting token nodes from a response

To map your LLM response to the collection of leaf nodes, use the `certus.interface` module:

```python
>>> import certus as ct
>>> from google.genai import types
>>> 
>>> data = "certus"
>>> logprobs = types.LogprobsResult(  # taken from `response.candidates[0].logprobsResult`
...     chosen_candidates=[
...         types.LogprobsResultCandidate(log_probability=0.0, token='"', token_id=24),
...         types.LogprobsResultCandidate(log_probability=-0.0123, token="certus", token_id=42),
...         types.LogprobsResultCandidate(log_probability=0.0, token='"', token_id=24),
...     ]
... )
>>> tokens = ct.interface.from_google(logprobs)
>>> tokens
[Token(value='"', logprob=0.0, start=0), Token(value='certus', logprob=-0.0123, start=1), Token(value='"', logprob=0.0, start=7)]

```

This list of token nodes is ready to be parsed into a tree.

### Building a tree

Consider this piece of JSON-friendly data:

```python
>>> import certus as ct
>>> 
>>> data = {
...     "name": "Henry Wilde",
...     "age": 29,
...     "longest_walk_km": 160.9,
...     "pets": [
...         {
...             "name": "Billie",
...             "species": "cat",
...             "favourite_foods": [
...                 "fish",
...                 "oat milk",
...                 {
...                     "name": "chicken",
...                     "preparation": "boiled",
...                     "when_sick": True,
...                 },
...             ],
...         },
...     ],
... }
>>> 

```

Let's say this data came from a `gpt-4o` response. We can tokenise this dictionary using `tiktoken` and simulate some
log-probabilities to go with them. From there, we can create a collection of `Token` leaf nodes ready for parsing;
details to do this are hidden below.

<details>
<summary>Simulating data tokens</summary>

```python
>>> import json
>>> import random
>>> 
>>> import tiktoken
>>> 
>>> def tokenise_string(string: str, encoder: tiktoken.Encoding) -> list[str]:
...     encoded = encoder.encode(string)
...     return [encoder.decode_single_token_bytes(e).decode() for e in encoded]
>>> 
>>> encoder = tiktoken.encoding_for_model("gpt-4o")
>>> data_tokenised = tokenise_string(json.dumps(data), encoder)
>>> 
>>> random.seed(0)
>>> tokens, position = [], 0
>>> for t in data_tokenised:
...     tokens.append(ct.nodes.Token(t, -round(random.expovariate(1e4), 6), position))
...     position += len(t)
>>> 
>>> assert json.loads("".join(t.value for t in tokens)) == data
>>> 

```

</details>
<br>

Now, we can parse this dictionary response and token nodes into a single `Object` node using the
`certus.parsers.parse_json()` function:

```python
>>> parsed = ct.parsers.parse_json(data, tokens)
>>> parsed  # doctest:+SKIP
Object(
    fields={
        'name': Composite(children=[Token(value=' "', logprob=-3e-05, start=8), Token(value='Henry', logprob=-7.2e-05, start=10), Token(value=' Wilde', logprob=-5.2e-05, start=15), Token(value='",', logprob=-0.000153, start=21)]), 
        'age': Token(value='29', logprob=-7e-05, start=31),
        'longest_walk_km': Composite(children=[Token(value='160', logprob=-0.000131, start=54), Token(value='.', logprob=-0.000229, start=57), Token(value='9', logprob=-0.000115, start=58)]),
        'pets': Array(
            elements=[
                Object(
                    fields={
                        'name': Composite(children=[Token(value=' "', logprob=-0.0002, start=78), Token(value='Bill', logprob=-3e-05, start=80), Token(value='ie', logprob=-0.000163, start=84), Token(value='",', logprob=-8e-05, start=86)]),
                        'species': Composite(children=[Token(value=' "', logprob=-0.000174, start=99), Token(value='cat', logprob=-0.00011, start=101), Token(value='",', logprob=-0.0, start=104)]),
                        'favourite_foods': Array(
                            elements=[
                                Composite(children=[Token(value=' ["', logprob=-8.4e-05, start=125), Token(value='fish', logprob=-2.7e-05, start=128), Token(value='",', logprob=-0.000343, start=132)]),
                                Composite(children=[Token(value=' "', logprob=-0.000163, start=134), Token(value='o', logprob=-5.9e-05, start=136), Token(value='at', logprob=-8e-06, start=137), Token(value=' milk', logprob=-3.9e-05, start=139), Token(value='",', logprob=-7.1e-05, start=144)]), 
                                Object(
                                    fields={
                                        'name': Composite(children=[Token(value=' "', logprob=-0.000123, start=155), Token(value='ch', logprob=-7.9e-05, start=157), Token(value='icken', logprob=-0.000168, start=159), Token(value='",', logprob=-7.8e-05, start=164)]),
                                        'preparation': Composite(children=[Token(value=' "', logprob=-9.1e-05, start=181), Token(value='bo', logprob=-4.9e-05, start=183), Token(value='iled', logprob=-8.6e-05, start=185), Token(value='",', logprob=-3.4e-05, start=189)]),
                                        'when_sick': Token(value=' true', logprob=-9e-06, start=204)
                                    }
                                )
                            ]
                        )
                    }    
                )
            ]
        )
    }
)

```

That's a lot of information, but you should be able to see a few node types here:

- `certus.nodes.core.Composite`: a collection of `Token` nodes
- `certus.nodes.struct.Array`: a collection of node elements, which behaves like a `list`
- `certus.nodes.struct.Object`: a mapping of keys to nodes, which behaves like a `dict`

We can leverage the `list`/`dict`-like properties of our `Object` node to look at the confidence in its various
components:

```python
>>> parsed.confidence  # the whole response
0.9999025047529705
>>> for key, value in parsed.items():
...     print(key.ljust(16), value.confidence)
name             0.9999232529452059
age              0.9999300024499428
longest_walk_km  0.9998416792007273
pets             0.9999055044649844
>>> 
>>> parsed["pets"][0]["favourite_foods"][-1]["name"].confidence  # Billie's last favourite food
0.9998880062717659

```
