#!/usr/bin/env python

import sys
import os.path

# Setup Paths
PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENTDIR)

from dsbox_dev_setup import path_setup
path_setup()

import grpc
import urllib
import pandas

import core_pb2 as core
import core_pb2_grpc as crpc

def run():
    channel = grpc.insecure_channel('localhost:8888')
    stub = crpc.CoreStub(channel)

    # Start Session
    session_response = stub.StartSession(
        core.SessionRequest(user_agent="xxx", version="1.0"))
    session_context = session_response.context
    print("Session started (%s)" % str(session_context.session_id))

    # Send pipeline creation request
    all_features = [
        core.Feature(
            data_uri="file:///tmp/data/o_185/data", feature_id="*")
    ]
    some_features = [
        core.Feature(data_uri="file:///tmp/data/o_185/data",
                   feature_id="d3mIndex"),
        core.Feature(data_uri="file:///tmp/data/o_185/data",
                   feature_id="Games_played"),
        core.Feature(
            data_uri="file:///tmp/data/o_185/data", feature_id="Runs"),
        core.Feature(
            data_uri="file:///tmp/data/o_185/data", feature_id="Hits"),
        core.Feature(data_uri="file:///tmp/data/o_185/data",
                   feature_id="Home_runs")
    ]
    target_features = [
        core.Feature(
            data_uri="file:///tmp/data/o_185/data", feature_id="*")
    ]
    task = core.TaskType.Value('CLASSIFICATION')
    task_subtype = core.TaskSubtype.Value('MULTICLASS')
    task_description = "Classify Hall of Fame"
    output = core.OutputType.Value('FILE')
    metrics = [core.Metric.Value('F1_MICRO')]
    max_pipelines = 20

    pipeline_ids = []

    '''
    print("Training with all features")
    pc_request = core.PipelineCreateRequest(
        context=session_context,
        train_features=all_features,
        task=task,
        task_subtype=task_subtype,
        task_description=task_description,
        output=output,
        metrics=metrics,
        target_features=target_features,
        max_pipelines=max_pipelines
    )

    # Iterate over results
    for pcr in stub.CreatePipelines(pc_request):
        print(str(pcr))
        if len(pcr.pipeline_info.scores) > 0:
            pipeline_ids.append(pcr.pipeline_id)
    '''

    print("Training with some features")
    pc_request = core.PipelineCreateRequest(
        context = session_context,
        train_features = some_features,
        task = task,
        task_subtype = task_subtype,
        task_description = task_description,
        output = output,
        metrics = metrics,
        target_features = target_features,
        max_pipelines = max_pipelines
    )

    # Iterate over results
    pipeline_id = None
    for pcr in stub.CreatePipelines(pc_request):
        print(str(pcr))
        if len(pcr.pipeline_info.scores) > 0:
            pipeline_ids.append(pcr.pipeline_id)

    # Execute pipelines
    for pipeline_id in pipeline_ids:
        print("Executing Pipeline %s" % pipeline_id)
        ep_request = core.PipelineExecuteRequest(
            context=session_context,
            pipeline_id=pipeline_id,
            predict_features=some_features
        )
        for ecr in stub.ExecutePipeline(ep_request):
            print(str(ecr))
            if len(ecr.result_uris) > 0:
                df = pandas.read_csv(ecr.result_uris[0], index_col="d3mIndex")
                print(df)

    stub.EndSession(session_context)

if __name__ == '__main__':
    run()
