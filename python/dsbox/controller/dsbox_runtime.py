import asyncio
import functools
import multiprocessing
import time
import typing
import uuid

from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor

class ProcessExecutorQueue:
    def __init__(self, loop: asyncio.AbstractEventLoop, *, 
                 max_workers: int = 0) -> None:

        self.loop = loop
        self._queue: asyncio.Queue = asyncio.Queue(loop=self.loop)

        # Set max number of subprocesses
        if max_workers == 0:
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max_workers

        # Use subprocess pool
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

        self._running_tasks: typing.List[asyncio.Task] = []

    def run(self, function: typing.Callable):
        coroutine = self.loop.run_in_executor(self._executor, function)
        task = self.loop.create_task(coroutine)
        task.add_done_callback(self._task_done)
        self._running_tasks.append(task)
        print('number of running tasks: ', len(self._running_tasks))
        return task

    async def _task_done(self, task):
        print('done call back start')
        self._running_tasks.remove(task)
        await self._queue.put(task)
        print('done call back end')

    def running_task_size(self) -> int:
        return len(self._running_tasks)

    def done_task_size(self) -> int:
        return self._queue.qsize()

    def all_task_size(self) -> int:
        return len(self._running_tasks) + self._queue.qsize() 

    async def get(self):
        return self._queue.get()

    def get_nowait(self) -> asyncio.Task:
        return self._queue.get_nowait()
