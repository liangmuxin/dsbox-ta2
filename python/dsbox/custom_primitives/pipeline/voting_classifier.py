from typing import List, Union
from itertools import zip_longest
from collections import Counter
import numpy as np

from d3m_metadata.sequence import ndarray
from .voting import VotingPrimitiveBase

Inputs = Union[List[ndarray], List[List[int]]]
Outputs = ndarray
Params = type(None)

class VotingClassifier(VotingPrimitiveBase[Inputs, Outputs, Params]):
    """
    Tally votes from other classifier outputs.
    """

    def __init__(self):
        params = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> Outputs:
        if not inputs:
            return None

        if len(inputs) > 1:
            # Input is list of 1d array. Each array is output of a classifier
            num_rows = inputs[0].shape[0]
            for array in inputs[1:]:
                assert array.shape[0] == num_rows, 'All votes must have the same number of rows'
    
            assert len(inputs[0].shape) == 1, 'Assume some type of label encoding'
            all_votes = zip_longest(inputs)
        else:
            # Input is a list of lists. Each sublist are votes for a single instance
            all_votes = inputs
        
        output = np.zeros(inputs[0].shape, dtype=inputs[0].dtype)
        for i, votes in enumerate(all_votes):
            c = Counter()
            c.update(votes)
            
            # To do: should break tie randomly
            [(val, _)] = c.most_common(1)
            output[i] = val

        return output

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        pass
