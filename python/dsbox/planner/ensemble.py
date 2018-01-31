import os
import sys
import os.path
import uuid
import copy
import math
import json
import numpy as np
import shutil
import traceback
import inspect
import importlib
import pandas as pd
import time

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.planner.common.pipeline import Pipeline, PipelineExecutionResult
from dsbox.planner.common.resource_manager import ResourceManager
from dsbox.planner.common.problem_manager import Metric, TaskType, TaskSubType

MIN_METRICS = [Metric.MEAN_SQUARED_ERROR, Metric.ROOT_MEAN_SQUARED_ERROR, Metric.ROOT_MEAN_SQUARED_ERROR_AVG, Metric.MEAN_ABSOLUTE_ERROR, Metric.EXECUTION_TIME]
DISCRETE_METRIC = [TaskSubType.BINARY, TaskSubType.MULTICLASS, TaskSubType.MULTILABEL, TaskSubType.OVERLAPPING, TaskSubType.NONOVERLAPPING]

class Ensemble(object):
    def __init__(self, problem, max_pipelines = 10):
        self.max_pipelines = max_pipelines
        self.predictions = None #predictions # Predictions dataframe
        self.metric_values =  None #metric_values # Dictionary of metric to value
        self.score = 0
        self.pipelines = []
        self.problem = problem
        self._analyze_metrics()

    def _analyze_metrics(self):
        # *** ONLY CONSIDERS 1 METRIC ***
        #self.minimize_metric = True if self.problem.metrics[0] in MIN_METRICS else False
        self.minimize_metric = []
        for i in range(0, len(self.problem.metrics)):
            print(self.problem.metrics[i])
            self.minimize_metric.append(True if self.problem.metrics[i] in MIN_METRICS else False)
        self.discrete_metric = True if self.problem.task_subtype in DISCRETE_METRIC else False


    def greedy_add(self, pipelines, X, y, max_pipelines = None):
        tic = time.time()
        if self.predictions is None:
            self.predictions = pd.DataFrame(index = X.index, columns = y.columns).fillna(0)     
            self.pipelines = []
    
        max_pipelines = self.max_pipelines if max_pipelines is None else max_pipelines
        #for j in range(to_add):
        found_improvement = True
        # change to unique pipelines?
        while found_improvement and len(np.unique([pl.id for pl in self.pipelines])) < max_pipelines:
            best_score =  float('inf') if self.minimize_metric else 0
            if self.metric_values is not None:
                best_metrics = self.metric_values

            found_improvement = False
            # first time through
            if not self.pipelines:
                best_predictions = pipelines[0].planner_result.predictions
                best_pipeline = pipelines[0]
                best_metrics = pipelines[0].planner_result.metric_values
                best_score = np.mean(np.array([a for a in best_metrics.values()]))
                found_improvement = True
                print('Best single pipeline score ',  str(best_score))
            else:
                for pipeline in pipelines:
                    metric_values = {}
                    #if type(self.predictions.values) :  
                    y_temp = (self.predictions.values * len(self.pipelines) + pipeline.planner_result.predictions.values) / (1.0*len(self.pipelines)+1)
                    #temp_predictions = (self.predictions[self.predictions.select_dtypes(include=['number']).columns] * len(self.pipelines) 
                    #                   + pipeline.predictions) / (len(self.pipelines)+1)

                    # check metric value binary or not
                    if self.discrete_metric:
                        y_rounded = np.rint(y_temp)
                    else:
                        y_rounded = y_temp
                    for i in range(0, len(self.problem.metrics)):
                        metric = self.problem.metrics[i]
                        fn = self.problem.metric_functions[i]
                        metric_val = self._call_function(fn, y, y_rounded)
                        if metric_val is None:
                            return None
                        metric_values[metric.name] = metric_val
                    score_improve = [v - best_metrics[k] for k, v in metric_values.items()]
                    score_improve = [score_improve[l] * (-1 if self.minimize_metric[l] else 1) for l in range(len(score_improve))]
                    score_improve = np.mean(np.array([a for a in score_improve]))
                    score = np.mean(np.array([a for a in metric_values.values()]))
                    
                    #print('Evaluating ', pipeline.primitives, score, score_improve)
                    if (score_improve > 0): # CHANGE TO > ?
                    #if (score > best_score and not self.minimize_metric) or (score < best_score and self.minimize_metric):
                        best_score = score
                        best_pipeline = pipeline
                        best_predictions = pd.DataFrame(y_temp, index = X.index, columns = y.columns)
                        best_metrics = metric_values
                        found_improvement = True
                    #pipelines.remove(pipeline)
            # evaluate / cross validate method?
            if found_improvement:
                self.pipelines.append(best_pipeline)
                self.predictions = best_predictions
                self.metric_values = best_metrics
                self.score = best_score
                #print('Adding ', best_pipeline.primitives, ' to ensemble of size ', str(len(self.pipelines)), '.  Ensemble Score: ', best_score)
            else:
                # END?
                #print (pipelines[0].planner_result.metric_values, pipelines[-1].planner_result.metric_values)
                print('Found ensemble of size ', str(len(self.pipelines)), ' with score ',  str(self.score))
                # TRYING TO ADD BEST PIPELINE: sorting is backwards if minimization metric

                # y_temp = (self.predictions.values * len(self.pipelines) + pipeline.planner_result.predictions.values) / (1.0*len(self.pipelines)+1)
                # if self.discrete_metric:
                #         y_rounded = np.rint(y_temp)
                # else:
                #         y_rounded = y_temp
                # metric_values = {}
                # for i in range(0, len(self.problem.metrics)):
                #     metric = self.problem.metrics[i]
                #     fn = self.problem.metric_functions[i]
                #     metric_val = self._call_function(fn, y, y_rounded)
                #     if metric_val is None:
                #         return None
                #     metric_values[metric.name] = metric_val
                
                
                # self.pipelines.append(pipelines[0] if )
                # self.predictions = pd.DataFrame(y_temp, index = X.index, columns = y.columns)
                # best_metrics = metric_values
                # self.metric_values = metric_values

                # print('Adding BEST metric.  Did NOT find improvement.  Score : ', np.mean(np.array([a for a in metric_values.values()])))

        ensemble_pipeline_ids = [pl.id for pl in self.pipelines]
        unique, indices, counts = np.unique(ensemble_pipeline_ids, return_index = True, return_counts = True) 
        self.unique_pipelines = [self.pipelines[index] for index in sorted(indices)]
        self.pipeline_weights = [counts[index] for index in list(np.argsort(indices))]
        if len(self.pipelines) < 10:
            print('Pipelines: ', self.pipelines) 
        print('Ensemble Unique: ', self.unique_pipelines, ' \n Counts: ', self.pipeline_weights)
        print('Ensemble Runtime ', time.time()-tic, ' with ', len(self.pipelines), ' pipelines')

    def _call_function(self, scoring_function, *args):
        mod = inspect.getmodule(scoring_function)
        try:
            module = importlib.import_module(mod.__name__)
            return scoring_function(*args)
        except Exception as e:
            sys.stderr.write("ERROR _call_function %s: %s\n" % (scoring_function, e))
            traceback.print_exc()
            return None