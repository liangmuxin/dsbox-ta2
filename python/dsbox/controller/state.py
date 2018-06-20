import enum
import collections
import typing
import uuid

class ProgressState(enum.Enum):
    # Default value. Not to be used.
    PROGRESS_UNKNOWN = 0

    # The process has been scheduled but is pending execution.
    PENDING = 1

    # The process is currently running. There can be multiple messages with this state
    # (while the process is running).
    RUNNING = 2

    # The process completed and final results are available.
    COMPLETED = 3

    # The process failed.
    ERRORED = 4


class SearchSolutionState:
    """
    Search request state

    Each search request has an unique id. Each search request has zero or more pipelines.
    """
    def __init__(self, search_id: str) -> None:
        if search_id is None:
            search_id = str(uuid.uuid4())
        self.search_id = search_id
        self._pipelines : typing.Dict[str, PipelineState] = {}
        self.stopped = False
        self.endded = False

class PipelineState:
    """
    Pipeline state

    Each pipeline has a unique id. Each pipeline has zero or more fitted pipelines.
    """
    def __init__(self, search_id: str, pipeline_id: str, pipeline, template_id: str = None) -> None:
        self.search_id = search_id
        self.pipeline_id = pipeline_id
        self.pipeline = pipeline
        self._fitted_pipeline : typing.List[FittedPipelineState] = []

class FittedPipelineState:
    """
    Fitted pipeline relation.

    Each fitted pipeline is associated with one pipeline, one dataset, and one runtime.
    """
    def __init__(self, pipeline_id: str, fitted_pipeline_id: str, *, dataset_id: str = None, runtime = None, progress = ProgressState.PROGRESS_UNKNOWN) -> None:
        self.pipeline_id = pipeline_id
        self.fitted_pipeline_id = fitted_pipeline_id
        self.dataset_id = dataset_id
        self.runtime = runtime
        self.progress = progress

    def update_progress(self, progress) -> None:
        self.progress = progress

class ProduceSolutionState:
    def __init__(self, fitted_pipeline_id, produce_id = None):
        if produce_id == None:
            produce_id =  str(uuid.uuid4())
        self.fitted_pipeline_id = fitted_pipeline_id
        self.produce_id = produce_id
        self.progress = ProgressStae.PROGRESS_UNKNOWN

class DsboxRuntimeState:
    """
    Manage data within a running Dsbox session.

    Knowns about all search requests, pipelines associated with each search request, and fitted_pipelines assoicated with each pipeline.
    """

    # Singleton class
    __instance = None
    def __new__(cls):
        if DsboxRuntimeState.__instance is None:
            DsboxRuntimeState.__instance = object.__new__(cls)
        return DsboxRuntimeState.__instance

    def __init__(self) -> None:
        # All known search requests
        self._search_request : typing.Dict[str, SearchSolutionState] = {}

        # All known pipelines (across all search request)
        self._pipeline : typing.Dict[str, PipelineState] = {}

        # All known fitted_pipelines (across all search request, all pipelines)
        self._fitted_pipeline : typing.Dict[str, FittedPipelineState] = {}

        # All known produce solution requests
        self._produce_request : typing.Dict[str, typing.List[ProduceSolutionState]] = {}

    def create_search_request(self):
        """
        Create a search request with genereated uuid
        """
        request = SearchSolutionState()
        self._search_request[request.search_id] = request
        return request

    def has_search_request(self, search_id: str) -> bool:
        return search_id in self._search_request

    def get_search_request(self, search_id: str) -> SearchSolutionState:
        """
        Returns search request state associated with the search_id
        """
        return self._search_request[search_id]

    def add_pipeline(self, search_id, pipeline_id, pipeline, *, template_id = None) -> PipelineState:
        """
        Associate pipeline with search_id
        """
        request = self._search_request[search_id]
        pipeline = PipelineState(search_id, pipeline_id, pipeline, template_id)
        request._pipelines[pipeline_id] = pipeline
        self._pipeline[pipeline_id] = pipeline
        return pipeline

    def has_pipeline(self, pipeline_id: str):
        """
        Returns true if pipeline_id is known
        """
        return pipeline_id in self._pipeline

    def get_pipeline(self, pipeline_id: str):
        """
        Returns pipeline state associated with the pipeline_id
        """
        return self._pipeline[pipeline_id]

    def add_fitted_pipeline(self, pipeline_id, fitted_pipeline_id, dataset_id = None, *, 
                            runtime = None, progress = ProgressState.PROGRESS_UNKNOWN) -> FittedPipelineState:
        """
        Assoicate fitted_pipeline with pipeline
        """
        pipeline = self._pipeline[pipeline_id]
        fitted_pipeline = FittedPipelineState(pipeline_id, fitted_pipeline_id, dataset_id = dataset_id, runtime = runtime, progress = progress)
        pipeline._fitted_pipeline.append(fitted_pipeline)
        self._fitted_pipeline[fitted_pipeline_id] = fitted_pipeline
        return fitted_pipeline

    def has_fitted_pipeline(self, fitted_pipeline_id):
        return fitted_pipeline_id in self._fitted_pipeline
        
    def get_fitted_pipeline(self, fitted_pipeline_id):
        return self._fitted_pipeline[fitted_pipeline_id]

    def create_produce_solution_request(self, fitted_pipeline_id):
        state = ProduceSolutionState(fitted_pipeline_id)
        self._produce_request[state.produce_id] = state
        return state
        
