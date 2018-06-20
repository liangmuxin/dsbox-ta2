import asyncio
import functools
import multiprocessing
import time
import typing
import uuid

from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor

from d3m.container.dataset import Dataset
from d3m.metadata.pipeline import Pipeline
from dsbox.template.runtime import Runtime

from dsbox.pipeline.fitted_pipeline import FittedPipeline

from .state import DsboxRuntimeState, ProgressState

class DsboxRuntime:
    """
    DSBox Runtime.

    Attributes
    ----------
    loop : AbstractEventLoop
        Python asyncio event loop
    runtime : Runtime
        Class to run pipelines

    """
    def __init__(self, loop: asyncio.AbstractEventLoop, runtime,
                 *, max_workers : int = 0, max_time_per_pipeline = None, max_time_per_produce=None) -> None:
        self.loop = loop
        self.runtime = runtime

        # Set max number of subprocesses
        if max_workers == 0:
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max_workers

        self.max_time_per_pipeline = max_time_per_pipeline
        self.max_time_per_produce = max_time_per_produce

        self.session = DsboxRuntimeState()
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

        self.tasks_by_search_id: typing.Dict[str, typing.List[Future]] = defaultdict(list)
        self.tasks_by_fitted_solution_id: typing.Dict[str, typing.List[Future]] = defaultdict(list)
        self.debug = True
        
    def fit_pipeline(self, search_id: str, pipeline: Pipeline, dataset: Dataset) -> FittedPipeline:
        """
        Train pipeline on dataset. This method blocks until the pipeline is finished.
        """
        
        if not self.session.has_pipeline(pipeline.id):
            self.session.add_pipeline(search_id, pipeline.id, pipeline)
        pipeline_relation = self.session.get_pipeline(pipeline.id)

        if dataset is not None:
            dataset_id = dataset.metadata.query(())['id']
        else:
            dataset_id = 'None'

        try:

            fitted_pipeline = FittedPipeline(pipeline, self.runtime(pipeline), dataset)
            fitted_pipeline_handle = self.session.add_fitted_pipeline(
                pipeline.id, fitted_pipeline.id, dataset_id, progress=ProgressState.PENDING)

            task = self.loop.create_task(self._fit_pipeline(search_id, fitted_pipeline, dataset))
            self.loop.run_until_complete(task)

            fitted_pipeline_handle.update_progress(ProgressState.COMPLETED)
            return fitted_pipeline
        except Exception: 
            fitted_pipeline_handle.update_progress(ProgressState.ERRORED)
        return None

    def fit_pipelines(self, search_id: str, pipelines: typing.List[Pipeline], dataset: Dataset) -> typing.List[FittedPipeline]:
        """
        Train multiple pipelines on dataset. This method blocks until all pipelines are finished. Each pipeline is ran under a subprocess.
        """
        
        if dataset is not None:
            dataset_id = dataset.metadata.query(())['id']
        else:
            dataset_id = 'None'

        fitted_pipelines = []
        fitted_pipeline_states = []
        tasks = []
        for pipeline in pipelines:
            self.session.add_pipeline(search_id, pipeline.id, pipeline)
            fitted_pipeline = FittedPipeline(pipeline, self.runtime(pipeline), dataset)
            fitted_pipelines.append(fitted_pipeline)
            fitted_pipeline_handle = self.session.add_fitted_pipeline(
                pipeline.id, fitted_pipeline.id, dataset_id, progress=ProgressState.PENDING)
            fitted_pipeline_states.append(fitted_pipeline_handle)
            task = self.loop.create_task(self._fit_pipeline(search_id, fitted_pipeline, dataset))
            tasks.append(task)

        # Run all tasks
        self.loop.run_until_complete(asyncio.gather(*tasks))
        
        return fitted_pipeline

    def produce_pipeline(self, fitted_solution_id: str, fitted_pipeline: FittedPipeline, dataset: Dataset) -> FittedPipeline:
        """
        Produce predictions on a fitted pipeline. This method blocks until produce is finished.
        """
        
        # fitted_pipeline already failed during fit stage
        if not fitted_pipeline.fitted or fitted_pipeline.cancel() or fitted.exception is not None:
            return fitted_pipeline

        # TODO: Do we need to update DsboxRuntimeState?
        
        try:
            task = self.loop.create_task(_produce_pipeline(self, fitted_solution_id, fitted_pipeline, dataset))
            self.loop.run_until_complete(task)
        except Exception:
            fitted_pipeline_handle.update_progress(ProgressState.ERRORED)

        return fitted_pipeline

    def produce_pipelines(self, fitted_solution_id: str, fitted_pipelines: typing.List[FittedPipeline], dataset: Dataset) -> typing.List[FittedPipeline]:
        """
        Produce predictions on a fitted pipeline. This method blocks until produce is finished.
        """

        # TODO: Do we need to update DsboxRuntimeState?

        valid_pipelines = []
        tasks = []
        for fitted_pipeline in fitted_pipelines:
            if not fitted_pipeline.fitted or fitted_pipeline.cancel() or fitted.exception is not None:
                valid_pipelines.append(fitted_pipeline)
                task = self.loop.create_task(self._produce_pipeline(fitted_solution_id, fitted_pipeline, dataset))
                tasks.append(task)

        # Run all task
        self.loop.run_until_complete(asyncio.gather(*tasks))

        return fitted_pipelines

            
    def end_search(self, search_id):
        """
        End all search realted to search_id, and release all associated resources.
        """
        if search_id in self.tasks_by_search_id:
            self.stop_search(search_id)
            del self.tasks_by_search_id[search_id]

    def stop_search(self, search_id):
        """
        Stop all searches related to search_id, but leaves all currently found solutions available.
        """
        if self.debug:
            print('stop_search {}'.format(search_id))
        if search_id in self.tasks_by_search_id:
            for task in self.tasks_by_search_id[search_id]:
                if self.debug:
                    print('  cancelling task', task)
                task.cancel()
        else:
            print('stop_search id not found {}'.format(search_id))
    

    async def _fit_pipeline(self, search_id: str, fitted_pipeline: FittedPipeline, dataset: Dataset) -> None:
        
        # use functools.partial because need to pass in keyword arguments
        subprocess_task = self.loop.run_in_executor(self.executor, functools.partial(fitted_pipeline.fit, inputs=[dataset]))

        self.tasks_by_search_id[search_id].append(subprocess_task)

        try:
            await asyncio.wait_for(subprocess_task, timeout=self.max_time_per_pipeline)
        except asyncio.TimeoutError:
            print('_fit_pipeline timeout {}:'.format(fitted_pipeline.id), end='')
            print(' done={} cancelled={}'.format(subprocess_task.done(), subprocess_task.cancelled()), end='')
        except asyncio.CancelledError:
            # Note: cancelling an executor task does NOT stop its active subprocess
            if self.debug:
                print('_fit_pipeline cancelled '.format(fitted_pipeline.id))

            # Pass back cancel status through fitted_pipeline
            fitted_pipeline.cancel()
        except Exception as e:
            if self.debug:
                print('_fit_pipeline {}: pipeline generated exception: {}'.format(fitted_pipeline.id, type(e)))
                print(' done={} cancelled={}'.format(subprocess_task.done(), subprocess_task.cancelled()))
            # Pass back exception through fitted_pipeline
            fitted_pipeline.set_exception(e)
        else:
            if self.debug:
                print('_fit_pipeline done={} cancelled={}'.format(subprocess_task.done(), subprocess_task.cancelled()))
        finally:
            self.tasks_by_search_id[search_id].remove(subprocess_task)
            
    async def _produce_pipeline(self, fitted_solution_id: str, fitted_pipeline: FittedPipeline, dataset: Dataset):

        subprocess_task = self.loop.run_in_executor(self.executor, functools.partial(fitted_pipeline.fit, inputs=[dataset]))

        self.tasks_by_fitted_solution_id[fitted_solution_id].append(subprocess_task)

        try:
            await asyncio.wait_for(subprocess_task, timeout=self.max_time_per_produce)
        except asyncio.TimeoutError:
            print('_produce_pipeline timeout {}:'.format(fitted_pipeline.id), end='')
            print(' done={} cancelled={}'.format(subprocess_task.done(), subprocess_task.cancelled()), end='')
        except asyncio.CancelledError:
            # Note: cancelling an executor task does NOT stop its active subprocess
            if self.debug:
                print('_produce_pipeline cancelled '.format(fitted_pipeline.id))

            # Should we have a separate produce cancel state?
            fitted_pipeline.cancel()
        except Exception as e:
            if self.debug:
                print('_produce_pipeline {}: pipeline generated exception: {}'.format(fitted_pipeline.id, type(e)))
                print(' done={} cancelled={}'.format(subprocess_task.done(), subprocess_task.cancelled()))
            # Pass back exception through fitted_pipeline
            fitted_pipeline.set_exception(e)
        else:
            if self.debug:
                print('_fit_pipeline done={} cancelled={}'.format(subprocess_task.done(), subprocess_task.cancelled()))
        finally:
            self.tasks_by_fitted_solution_id[search_id].remove(subprocess_task)
