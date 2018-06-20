import json 
import pickle
import typing
import uuid

from d3m.exceptions import InvalidStateError
from d3m.container.dataset import Dataset
from d3m.metadata.pipeline import Pipeline
from d3m.runtime import Runtime

from dsbox.template.search import ConfigurationSpace, ConfigurationPoint

# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)
TP = typing.TypeVar('TP', bound='FittedPipeline')

class FittedPipeline:
    """
    Fitted pipeline is a pipeline with its associated Runtime.

    Attributes
    ----------
    pipeline: Pipeline
        a pipeline
    dataset: Dataset
        identifier for a dataset
    runtime: Runtime
        runtime containing fitted primitives
    id: str
        the id of the pipeline
    folder_loc: str
        the location of the files of pipeline
    """

    def __init__(self, pipeline = None, runtime = None, dataset = None, *, id = None):
        self.dataset = dataset
        self.runtime = runtime
        self.pipeline = pipeline
        if id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = id
        self.folder_loc = ''
        self.fitted = False
        self._exception = None
        self._cancelled = False

    def fit(self, **arguments) -> None:
        self.runtime.fit(**arguments)
        self.fitted = True

    def produce(self, **arguments):
        if not self.fitted:
            raise InvalidStateError('FittedPipeline: Running produce before fit ({})'.format(self.id))
        return self.runtime.produce(**arguments)

    def set_exception(self, exception):
        self._exception = exception

    def exception(self):
        return self._exception

    def cancel(self):
        self._cancelled = True

    def cancelled(self):
        return self._cancelled

    # @classmethod
    # def create(cls:typing.Type[TP], configuration: ConfigurationPoint[PythonPath], dataset: Dataset) -> TP:
    #     '''
    #     Initialize the FittedPipeline with the configurations
    #     '''
    #     pipeline_to_load = configuration.data['pipeline']
    #     run = configuration.data['runtime']
    #     fitted_pipeline_loaded = cls(pipeline_to_load, run, dataset)
    #     return fitted_pipeline_loaded

    def save(self, folder_loc : str) -> None:
        '''
        Save the given fitted pipeline from TemplateDimensionalSearch
        '''
        self.folder_loc = folder_loc
        print("The pipeline files will be stored in:")
        print(self.folder_loc)

        # save the pipeline with json format
        json_loc = self.folder_loc + '/pipelines/' + self.id + '.json'
        with open(json_loc, 'w') as f:
            self.pipeline.to_json_content(f)

        # save the pickle files of each primitive step
        pkl_loc = self.folder_loc + '/excutables/' + self.id
        for i in range(0, len(self.runtime.execution_order)):
            print("Now saving step_", i)
            n_step = self.runtime.execution_order[i]
            each_step = self.runtime.pipeline[n_step]
            '''
            NOTICE:
            runing both of get_params and hyperparams will cause the error of 
            "AttributeError: 'RandomForestClassifier' object has no attribute 'oob_score_'"
            print(each_primitive.get_params())
            print(each_step.hyperparams)
            '''
            file_loc = pkl_loc + "_step_" + str(i) + ".pkl"
            with open(file_loc, "wb") as f:
                pickle.dump(each_step, f)

    @classmethod
    def load(cls:typing.Type[TP], folder_loc: str, pipeline_id: str, dataset: Dataset) -> TP:
        '''
        Load the pipeline with given pipeline id and folder location
        '''
        # load pipeline from json
        json_loc = folder_loc + '/pipelines/' + pipeline_id + '.json'
        print("The following pipeline files will be loaded:")
        print(json_loc)
        with open(json_loc, 'r') as f:
            pipeline_to_load = Pipeline.from_json_content(f)

        # load detail fitted parameters from pkl files
        run = Runtime(pipeline_to_load)
        pkl_loc = folder_loc + '/excutables/' + pipeline_id
        for i in range(0, len(run.execution_order)):
            print("Now loading step_", i)
            n_step = run.execution_order[i]
            file_loc = pkl_loc + "_step_" + str(i) + ".pkl"
            with open(file_loc, "rb") as f:
                each_step = pickle.load(f)
                run.pipeline[n_step] = each_step

        fitted_pipeline_loaded = cls(pipeline_to_load, run, dataset)
        return fitted_pipeline_loaded

