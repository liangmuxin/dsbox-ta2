from primitive_interfaces.featurization import FeaturizationPrimitiveBase, FeaturizationTransformerPrimitiveBase
from primitive_interfaces.base import Inputs, Outputs, Params, Hyperparams,\
    PrimitiveBase
from typing import NamedTuple
from d3m_metadata.sequence.numpy import ndarray
from dsbox.custom_primitives.pipeline.voting import VotingPrimitiveBase

class SplitFeaturizationPrimitiveBase(FeaturizationPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which transform raw data into a more usable form.

    This class differ from FeaturizationPrimitiveBase in that each row
    of the input may correspond to one or more rows in the output.

    Use this version for featurizers that allow for fitting (for domain-adaptation, data-specific deep
    learning, etc.).  Otherwise use `MultiFeaturizationTransformerPrimitiveBase`.
    """

class SplitFeaturizationTransformerPrimitiveBase(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A base class for primitives which transform raw data into a more usable form.

    This class differ from FeaturizationPrimitiveBase in that each row
    of the input may correspond to one or more rows in the output.

    Use this version for featurizers that do not require or allow any fitting, and simply
    transform data on demand.  Otherwise use `MutliFeaturizationPrimitiveBase`.
    """

MergeInputs = NamedTuple(['FeaturizationVotes', 
                          ('votes', ndarray), 
                          ('grouping', ndarray),
                          ('voting_primitive', VotingPrimitiveBase)])
MergeOutputs = ndarray
class MergePrimitiveBase(PrimitiveBase[MergeInputs, Outputs, Params, Hyperparams]):
    """
    Merge classification/regression results from split features
    """

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> Outputs:
        merge_list = []
        for start, end in zip(inputs.grouping, inputs.grouping[1:]):
            merge_list.append(inputs.votes[start:end])
        return inputs.voting_primitive(merge_list)
    
    
    
