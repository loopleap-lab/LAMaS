"""Test helpers: a controller that always picks the same workflow."""

from typing import List, Optional

from lamas import Controller, Policy, Selection, StopPolicy


class Fixed(Controller):
    name = "fixed"

    def __init__(self, layers: List[List[str]]):
        self.layers = layers

    def select(self, query, operators, *, seed=None, track_grad=False) -> Selection:
        return Selection(layers=[list(layer) for layer in self.layers])


def fixed_policy(layers: List[List[str]], stop: Optional[StopPolicy] = None) -> Policy:
    return Policy(name="test", controller=Fixed(layers), stop=stop or StopPolicy(), task="general",
                  operators=sorted({name for layer in layers for name in layer}))


WIDE = [["GenerateCoT", "GenerateCoT", "MultiGenerateCoT"], ["ScEnsemble"]]          # three in parallel, then a merge
CHAIN = [["GenerateCoT"], ["SelfRefine"], ["SelfRefine"], ["SelfRefine"]]           # each waits for the one before
