from typing import List, Union
from itertools import zip_longest
import numpy as np

from d3m_metadata.sequence import ndarray
from .voting import VotingPrimitiveBase
from d3m_metadata.hyperparams import Choice

Inputs = Union[List[ndarray], List[List[float]]]
Outputs = ndarray
Params = type(None)
Hyperparameter = Choice(choices=['mean', 'median', 'max', 'min'],
                        default='mean')

class VotingRegressor(VotingPrimitiveBase[Inputs, Outputs, Params, Hyperparameter]):
    """
    Tally votes from other classifier outputs.
    """

    def __init__(self, *, hyperparameter: Hyperparameter):
        self.params = None
        self.hyperparameter = hyperparameter
        if self.hyperparameter == 'mean':
            self.func = np.mean
        elif self.hyperparameter == 'median':
            self.func = np.median
        elif self.hyperparameter == 'max':
            self.func = np.max
        elif self.hyperparameter == 'median':
            self.func = np.min
        else:
            self.func = np.mean

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> Outputs:
        if not inputs:
            return None

        if len(inputs) > 1:
            # Input is list of 1d array. Each array is output of a regressor
            num_rows = inputs[0].shape[0]
            for array in inputs[1:]:
                assert array.shape[0] == num_rows, 'All votes must have the same number of rows'
    
            assert len(inputs[0].shape) == 1, 'Assume some type of label encoding'
    
            output = [self.func(votes) for votes in zip_longest(inputs)]
        else:
            # Input is a list of lists. Each sublist are votes for a single instance 
            output = [self.func(votes) for votes in inputs]
        output = np.asanyarray(output)
        return output

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        pass