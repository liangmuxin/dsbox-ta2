import asyncio
import functools
import multiprocessing
import time
import typing
import uuid

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

class ProcessExecutorQueue:
    """
    This class executes `Callable` and stores results in asyncio.Queue.

    Attributes
    ----------
    loop : asyncio.AbstractEventLoop
        An asyncio.AbstractEventLoop in `run_forever` mode
    max_workers: int
        Maximum of concurrent subprocesses to use
    """
    def __init__(self, loop: asyncio.AbstractEventLoop, *, 
                 max_workers: int = 0) -> None:

        self.loop = loop

        # Set max number of subprocesses
        if max_workers == 0:
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max_workers

        # Use subprocess pool
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

        # List of running tasks, i.e. subprocesses
        self._running_tasks: typing.List[asyncio.Task] = []

        # Queue of completed tasks
        self._finished_queue: asyncio.Queue = asyncio.Queue(loop=self.loop)


    def run(self, function: typing.Callable, *args) -> asyncio.Task:
        """
        Call the `function` in a subprocess. Return the results in a asyncio.Task by
        wraping the coroutine assoicated with that subprocess.

        """
        coroutine = self.loop.run_in_executor(self._executor, function, *args)

        # If an exception occurs in the coroutine, calling task.result() will cause python
        # to raise that exception. To avoid raising that exception use task.exception()
        # method to see that exception.

        task = self.loop.create_task(self._run(coroutine))
        
        task.add_done_callback(self._run_done)
        self._running_tasks.append(task)
        print('number of running tasks: ', len(self._running_tasks))
        return task
        
    async def _run(self, coroutine):
        result = await coroutine
        return result

    def _run_done(self, task):
        """
        Once a task is done, remove it from the self._running_tasks list and add it to the
        finsihed queue.
        """
        # print('done call back start')
        self._running_tasks.remove(task)
        self._finished_queue.put_nowait(task)
        # print('  task: done={} cancelled={}'.format(task.done(), task.cancelled()))
        # if task.done():
        #     if task.exception() is not None:
        #         print('  exception', task.exception())
        #     else:
        #         print('done call back end. result=', task.result())

    def has_more_tasks(self) -> bool:
        """
        Return True if there are either tasks running or more tasks in the finished queue
        """
        return self.all_task_size() > 0

    def has_more_finished_task(self) -> bool:
        return self._finished_queue.qsize() > 0
    
    def running_task_size(self) -> int:
        return len(self._running_tasks)

    def finished_task_size(self) -> int:
        return self._finished_queue.qsize()

    def all_task_size(self) -> int:
        return len(self._running_tasks) + self._finished_queue.qsize()

    def get(self):
        """
        Returns a coroutine to remove and return a finished task from the queue.
        """
        return self._finished_queue.get()

    def get_nowait(self) -> asyncio.Task:
        """
        Remove and return a finished task from the queue, else raise QueueEmpty
        """
        return self._finished_queue.get_nowait()
