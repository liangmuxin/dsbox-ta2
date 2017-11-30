from typing import List
from primitive_interfaces.base import PrimitiveBase, Params, Hyperparams
from d3m_metadata.sequence import ndarray

Inputs = List[ndarray]
Outputs = ndarray

class VotingPrimitiveBase(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Base class for primitves tallying votes from other classifier outputs.
    """
