import json
import glob
import typing

from d3m import index
from d3m.container.dataset import SEMANTIC_TYPES
from d3m.metadata.problem import TaskType, TaskSubtype

from dsbox.template.template import TemplatePipeline, DSBoxTemplate


class TemplateDescription:
    """
    Description of templates in the template library.

    Attributes
    ----------
    task : TaskType
        The type task the template handles
    template: TemplatePipeline
        The actual template
    target_step: int
        The step of the template that extract the ground truth target from
        the dataset
    predicted_target_step: int
        The step of the template generates the predictions
    """

    def __init__(self, task: TaskType, template: TemplatePipeline,
                 target_step: int, predicted_target_step: int) -> None:
        self.task = task
        self.template = template

        # Instead of having these attributes here, probably should attach
        # attributes to the template steps
        self.target_step = target_step
        self.predicted_target_step = predicted_target_step


class TemplateLibrary:
    """
    Library of template pipelines
    """

    def __init__(self, library_dir: str = None) -> None:
        self.templates: typing.List[typing.Type[DSBoxTemplate]] = []
        self.primitive: typing.Dict = index.search()

        self.library_dir = library_dir
        if self.library_dir is None:
            self._load_library()

        self._load_inline_templates()

    def get_templates(self, task: TaskType, subtype: TaskSubtype, taskSourceType: SEMANTIC_TYPES) -> typing.List[DSBoxTemplate]:
        results = []
        for template_class in self.templates:
            template = template_class()
            #import pdb
            #pdb.set_trace()

            if task.name in template.template['taskType'] and (subtype==TaskSubtype.NONE or subtype.name in template.template['taskSubtype']):
                # if there is only one task source type which is table, we don't need to check other things
                if taskSourceType == {"table"} and template.template['inputType'] == "table":
                    results.append(template)
                else:
                    # otherwise, we need to process in another way because "table" source type exist nearly in every dataset
                    if "table" in taskSourceType:
                        taskSourceType.remove("table")

                    for each_source_type in taskSourceType:
                        if each_source_type in {template.template['inputType']}:
                            results.append(template)
        # if we finally did not find a proper template to use
        if results == []:
            print("Error: Can't find a suitable template type to fit the problem.")
        else:
            print("[INFO] Template choices:")
        # otherwise print the template list we added
            for each_template in results:
                print("Template '", each_template.template["name"], "' has been added to template base.")
        return results

    def _load_library(self):
        # TODO
        # os.path.join(library_dir, 'template_library.yaml')
        pass

    def _load_inline_templates(self):
        # These 2 are in old version style, do not load them!
        # self.templates.append(self._generate_simple_classifer_template())
        # self.templates.append(self._generate_simple_regressor_template())

        # added new inline_templates muxin
        self.templates.append(DefaultRegressionTemplate)
        self.templates.append(DefaultClassificationTemplate)
        self.templates.append(DefaultTimeseriesCollectionTemplate)
        self.templates.append(DefaultImageProcessingRegressionTemplate)
        self.templates.append(DefaultGraphMatchingTemplate)
        #self.templates.append(DoesNotMatchTemplate2)

class SemanticTypeDict(object):
    def __init__(self, libdir):
        self.pos = libdir
        self.mapper = {}

    def read_primitives(self) -> None:

        # jsonPath = os.path.join(libdir, filename)
        # print(self.pos)
        user_Defined_Confs = glob.glob(
            self.pos + "/*_template_semantic_mapping.json")
        # print(user_Defined_Confs)
        for u in user_Defined_Confs:
            with open(u, "r") as cf:
                print("opened", u)
                for v in json.load(cf).items():
                    self.mapper[v[0]] = v[1]

    def create_configuration_space(self, template: TemplatePipeline):
        definition = {}
        # for t in TemplatePipeline:
        #     if isinstance(t, list):
        steps = template.template_nodes.keys()
        for s in steps:
            if template.template_nodes[s].semantic_type in self.mapper.keys():
                definition[s] = self.mapper[
                    template.template_nodes[s].semantic_type]

        # return SimpleConfigurationSpace(definition)
        return definition


# class DoesNotMatchTemplate1(DSBoxTemplate):
#     def __init__(self):
#         DSBoxTemplate.__init__(self)
#         self.template = {
#             "name": "Does_Not_Match_template1",
#             "taskType": TaskType.CLASSIFICATION.name,  # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
#             "taskSubtype" : {TaskSubtype.BINARY.name,TaskSubtype.MULTICLASS.name},
#             "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
#             "output": "model_step",  # Name of the final step generating the prediction
#             "target": "extract_target_step",  # Name of the step generating the ground truth
#             "steps": [
#
#                 {
#                     "name": "denormalize_step",
#                     "primitives": ["d3m.primitives.dsbox.Denormalize"],
#                     "inputs": ["template_input"]
#                 },
#                 {
#                     "name": "to_dataframe_step",
#                     "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
#                     "inputs": ["denormalize_step"]
#                 },
#                 {
#                     "name": "column_parser_step",
#                     "primitives": ["d3m.primitives.data.ColumnParser"],
#                     "inputs": ["to_dataframe_step"]
#                 },
#
#                 {
#                     "name": "extract_attribute_step",
#                     "primitives": ["d3m.primitives.data.ExtractAttributes"],
#                     "inputs": ["column_parser_step"]
#                 },
#                 {
#                     "name": "cast_1_step",
#                     "primitives": ["d3m.primitives.data.CastToType"],
#                     "inputs": ["extract_attribute_step"]
#                 },
#                 {
#                     "name": "extract_target_step",
#                     "primitives": ["d3m.primitives.data.ExtractTargets"],
#                     "inputs": ["column_parser_step"]
#                 },
#                 {
#                     "name": "cast_2_step",
#                     "primitives": ["d3m.primitives.data.CastToType"],
#                     "inputs": ["extract_target_step"]
#                 },
#                 {
#                     "name": "model_step",
#                     "primitives": ["d3m.primitives.common_primitives.RandomForestClassifier", "d3m.primitives.sklearn_wrap.SKSGDClassifier"],
#                     "inputs": ["cast_1_step", "cast_2_step"]
#                 }
#             ]
#         }
#
#     # @override
#     def importance(datset, problem_description):
#         return 7
#

# class DoesNotMatchTemplate2(DSBoxTemplate):
#     def __init__(self):
#         DSBoxTemplate.__init__(self)
#         self.template = {
#             "name": "Does_Not_Match_template2",
#             "taskType": TaskType.CLASSIFICATION.name,  # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
#             "taskSubtype" : {TaskSubtype.BINARY.name,TaskSubtype.MULTICLASS.name},
#             "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
#             "output": "model_step",  # Name of the final step generating the prediction
#             "target": "extract_target_step",  # Name of the step generating the ground truth
#             "steps": [
#                 {
#                     "name": "denormalize_step",
#                     "primitives": ["d3m.primitives.dsbox.Denormalize"],
#                     "inputs": ["template_input"]
#                 },
#                 {
#                     "name": "to_dataframe_step",
#                     "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
#                     "inputs": ["denormalize_step"]
#                 },
#                 {
#                     "name": "column_parser_step",
#                     "primitives": ["d3m.primitives.data.ColumnParser"],
#                     "inputs": ["to_dataframe_step"]
#                 },
#                 {
#                     "name": "extract_attribute_step",
#                     "primitives": [{
#                         "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
#                         "hyperparameters":
#                             {
#                                 'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
#                                 'use_columns': (),
#                                 'exclude_columns': ()
#                             }
#                     }],
#                     "inputs": ["column_parser_step"]
#                 },
#                 {
#                     "name": "cast_1_step",
#                     "primitives": ["d3m.primitives.data.CastToType"],
#                     "inputs": ["extract_attribute_step"]
#                 },
#                 {
#                     "name": "extract_target_step",
#                     "primitives": [{
#                         "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
#                         "hyperparameters":
#                             {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
#                              'use_columns': (),
#                              'exclude_columns': ()
#                              }
#                     }],
#                     "inputs": ["column_parser_step"]
#                 },
#                 {
#                     "name": "cast_2_step",
#                     "primitives": ["d3m.primitives.data.CastToType"],
#                     "inputs": ["extract_target_step"]
#                 },
#                 {
#                     "name": "model_step",
#                     "primitives": ["d3m.primitives.sklearn_wrap.SKSGDClassifier","d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
#                     "inputs": ["cast_1_step", "cast_2_step"]
#                 }
#             ]
#         }
#     # @override
#     def importance(datset, problem_description):
#         return 7

# class SerbansClassificationTemplate(DSBoxTemplate):
#     def __init__(self):
#         DSBoxTemplate.__init__(self)
#         self.template = {
#             "name": "Serbans_classification_template",
#             "taskType": TaskType.CLASSIFICATION.name,
#             "inputType": "table",
#             "taskSubtype" : {TaskSubtype.BINARY.name,TaskSubtype.MULTICLASS.name},
#             "output": "model_step",
#             "target": "extract_target_step",
#             "steps": [
#
#                 # join several tables into one
#                 {
#                     "name": "denormalize_step",
#                     "primitives": ["d3m.primitives.dsbox.Denormalize"],
#                     "inputs": ["template_input"]
#                 },
#
#                 # create a DF
#                 {
#                     "name": "to_dataframe_step",
#                     "primitives": [
#                         "d3m.primitives.datasets.DatasetToDataFrame"],
#                     "inputs": ["denormalize_step"]
#                 },
#
#                 #
#                 {
#                     "name": "column_parser_step",
#                     "primitives": ["d3m.primitives.data.ColumnParser"],
#                     "inputs": ["to_dataframe_step"]
#                 },
#
#                 # extract columns with the 'attribute' metadata
#                 {
#                     "name": "extract_attribute_step",
#                     "primitives": ["d3m.primitives.data.ExtractAttributes"],
#                     "inputs": ["column_parser_step"]
#                 },
#
#                 # change strings to correct column type
#                 {
#                     "name": "cast_1_step",
#                     "primitives": ["d3m.primitives.data.CastToType"],
#                     "inputs": ["extract_attribute_step"]
#                 },
#
#                 {
#                     "name": "impute_step",
#                     "primitives": [
#                         {
#                             "primitive":
#                                 "d3m.primitives.sklearn_wrap.SKImputer",
#                             # "hyperparameters": {
#                             #     "strategy": ["mean", "median", "most_frequent"],
#                             # },
#                         },
#                     ],
#                     "inputs": ["cast_1_step"]
#                 },
#                 # {
#                 #     "name": "standardize",
#                 #     "primitives": ["dsbox.datapreprocessing.cleaner.IQRScaler"], # FIXME: want d3m name
#                 #     "inputs": ["impute_step"]
#                 # },
#
#                 # processing target column
#                 {
#                     "name": "extract_target_step",
#                     "primitives": ["d3m.primitives.data.ExtractTargets"],
#                     "inputs": ["column_parser_step"]
#                 },
#                 {
#                     "name": "cast_2_step",
#                     "primitives": ["d3m.primitives.data.CastToType"],
#                     "inputs": ["extract_target_step"]
#                 },
#
#                 # running a primitive
#                 {
#                     "name": "model_step",
#                     "primitives": [
#                         {
#                             "primitive":
#                                 "d3m.primitives.sklearn_wrap." +
#                                  "SKGradientBoostingClassifier",
#                             "hyperparameters": {
#                                 "n_estimators": [50,75,100],
#                                     },
#                         },
#                         # {
#                         #     "primitive":
#                         #         "d3m.primitives.sklearn_wrap.SKSGDClassifier",
#                         #     "hyperparameters": {}
#                         # },
#                     ],
#                     # attributes (output of impute_step) and target (output
#                     # of casting method)
#                     "inputs": ["impute_step", "cast_2_step"]
#                 }
#             ]
#         }
#
#     # @override
#     def importance(datset, problem_description):
#         return 7

class DefaultClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_classification_template",
            "taskSubtype" : {TaskSubtype.BINARY.name,TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,  # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "column_parser_step",
                    "primitives": ["d3m.primitives.data.ColumnParser"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "cast_1_step",
                    "primitives": ["d3m.primitives.data.CastToType"],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["cast_1_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    # "runtime": {
                    #     "cross_validation": 10,
                    #     "stratified": True
                    # },
                    "primitives": [{
                        "primitive": "d3m.primitives.sklearn_wrap.SKRandomForestClassifier"
                        }],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }
    # @override
    def importance(datset, problem_description):
        return 7

class DefaultRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_regression_template",
            "taskType": TaskType.REGRESSION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING',
            # 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION',
            # 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype" : {TaskSubtype.UNIVARIATE.name,TaskSubtype.MULTIVARIATE.name},
            "inputType": "table",
            # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",
            # Name of the final step generating the prediction
            "target": "extract_target_step",
            # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.datasets.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": [
                        "d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive":
                            "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "column_parser_step",
                    "primitives": ["d3m.primitives.data.ColumnParser"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "cast_1_step",
                    "primitives": ["d3m.primitives.data.CastToType"],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["cast_1_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive":
                            "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': (
                            'https://metadata.datadrivendiscovery.org/types'
                            '/Target',
                            'https://metadata.datadrivendiscovery.org/types'
                            '/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKARDRegression",
                        "hyperparameters":
                            {
                            }
                    },{
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKSGDRegressor",
                        "hyperparameters":
                            {
                            }
                    }, {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKGradientBoostingRegressor",
                        "hyperparameters":
                            {
                            }
                    }],
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": False
                    },
                    "inputs": ["impute_step", "extract_target_step"]
                },
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7

class DefaultTimeseriesCollectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_timeseries_collection_template",
            "taskType": TaskType.CLASSIFICATION.name, # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype" : {TaskSubtype.BINARY.name,TaskSubtype.MULTICLASS.name},
            "inputType": "timeseries",  # See SEMANTIC_TYPES.keys() for range of values
            "output" : "random_forest_step",  # Name of the final step generating the prediction
            "target" : "extract_target_step",  # Name of the step generating the ground truth
            "steps": [

                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive":"d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target','https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                            }
                        }],
                    "inputs": ["to_dataframe_step"]
                },
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data.ColumnParser"],
                #     "inputs": ["extract_target_step"]
                # },

                # read X value
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.dsbox.TimeseriesToList"],
                    "inputs": ["to_dataframe_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": ["d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization"],
                    "inputs": ["timeseries_to_list_step"]
                },

                {
                    "name": "random_forest_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestClassifier"],
                    "inputs": ["random_projection_step", "extract_target_step"]
                },
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class DefaultImageProcessingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_image_processing_regression_template",
            "taskType": TaskType.REGRESSION.name, # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype" : {TaskSubtype.UNIVARIATE.name,TaskSubtype.MULTIVARIATE.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output" : "regressor_step",  # Name of the final step generating the prediction
            "target" : "extract_target_step",  # Name of the step generating the ground truth
            "steps": [

                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive":"d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target','https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                            }
                        }],
                    "inputs": ["to_dataframe_step"]
                },
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data.ColumnParser"],
                #     "inputs": ["to_dataframe_step"]
                # },
                # read X value
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.dsbox.DataFrameToTensor"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    "primitives": ["d3m.primitives.dsbox.ResNet50ImageFeature"],
                    # or "primitives": ["d3m.primitives.dsbox.Vgg16ImageFeature"],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKPCA"],
                    "inputs": ["feature_extraction"]
                },

                {
                    "name": "regressor_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKRandomForestRegressor"],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
            ]
        }
    def importance(datset, problem_description):
        return 7


class DefaultGraphMatchingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_GraphMatching_Template",
            "taskType": TaskType.GRAPH_MATCHING.name,
             # for some special condition, the taskSubtype can be "NONE" which indicate no taskSubtype given
            "taskSubtype" : "NONE",
            "inputType": "graph",
            "output": "model_step",
            "steps": [
                {
                    "name": "model_step",
                    "primitives": ["d3m.primitives.sri.psl.GraphMatchingLinkPrediction"],
                    "inputs":["template_input"]
                }
            ]
        }
    def importance(dataset, problem_description):
        return 7
