import abc
from primitive_interfaces.featurization import FeaturizationPrimitiveBase, FeaturizationTransformerPrimitiveBase
from primitive_interfaces.base import Inputs, Outputs, Params, Hyperparams,\
    PrimitiveBase
from typing import NamedTuple
from d3m_metadata.sequence.numpy import ndarray
from dsbox.custom_primitives.pipeline.voting import VotingPrimitiveBase

class SplitMixin(Generic[Inputs, Outputs, Params, Hyperparams]):
    """
    This mixin provides information needed to map multiple split rows
    to the original row. Some featurization primitives may generate
    multiple rows for each input row.

    """
    @abc.abstractmethod
    def grouping_output(self, *) -> ndarray:
        """
        Returns the grouping of split rows.

        The MergePrimitiveBase uses this grouping to map the split
        rows to the original input row.

        """

class SplitFeaturizationPrimitiveBase(
        SplitMixin[Inputs, Outputs, Params, Hyperparams],
        FeaturizationPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which transform raw data into a more usable form.

    This class differ from FeaturizationPrimitiveBase in that each row
    of the input may correspond to one or more rows in the output.

    Use this version for featurizers that allow for fitting (for domain-adaptation, data-specific deep
    learning, etc.).  Otherwise use `MultiFeaturizationTransformerPrimitiveBase`.
    """

class SplitFeaturizationTransformerPrimitiveBase(
        SplitMixin[Inputs, Outputs, Params, Hyperparams],
        FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A base class for primitives which transform raw data into a more usable form.

    This class differ from FeaturizationPrimitiveBase in that each row
    of the input may correspond to one or more rows in the output.

    Use this version for featurizers that do not require or allow any fitting, and simply
    transform data on demand.  Otherwise use `MutliFeaturizationPrimitiveBase`.
    """

MergeInputs = NamedTuple(['FeaturizationVotes',
                          ('votes', ndarray),
                          ('split_primitive', SplitMixin),
                          ('voting_primitive', VotingPrimitiveBase)])
MergeOutputs = ndarray
class MergePrimitiveBase(PrimitiveBase[MergeInputs, MergeOutputs, Params, Hyperparams]):
    """
    Merge classification/regression results from split features
    """

    def produce(self, *, inputs: MergeInputs, timeout: float = None, iterations: int = None) -> MergeOutputs:
        grouping = inputs.split_primitive.grouping_output()
        voting_primitive = inputs.voting_primitive
        votes = inputs.votes
        merge_list = []
        for start, end in zip(grouping, grouping[1:]):
            merge_list.append(votes[start:end])
        return voting_primitive.produce(merge_list)
