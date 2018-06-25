import asyncio
import functools
import os
import sys
import time
import traceback

from concurrent.futures import Future, ProcessPoolExecutor

path = os.path.abspath('../..')
print('Add to python path: ', path)
sys.path.append(path)

from importlib import reload

import dsbox.controller.dsbox_runtime
reload(dsbox.controller.dsbox_runtime)
from dsbox.controller.dsbox_runtime import ProcessExecutorQueue

async def stop(loop):
    await asyncio.sleep(10)
    print('stop')
    loop.stop()

def run_pipeline(id, timeout=5):
    print('Pipeline {} start'.format(id))
    time.sleep(timeout)
    if timeout==2:
        raise ValueError('EXCEPTION run_pipeline {}'.format(id))
    print('PIPELINE {} done'.format(id))
    return ('pipeline {} done'.format(id))

async def run():
    print('Starting...')
    executor_queue = ProcessExecutorQueue(loop)
    t1 = executor_queue.run(functools.partial(run_pipeline, 1, timeout=3))
    t2 = executor_queue.run(functools.partial(run_pipeline, 2, timeout=2))
    t3 = executor_queue.run(functools.partial(run_pipeline, 3, timeout=1))

    t1.cancel()

    while executor_queue.has_more_tasks():
        task = await executor_queue.get()
        print('Got a pipeline result')
        try:
            if task.done() and not task.cancelled():
                if task.exception() is None:
                    print('  Result = {}'.format(task.result()))
                else:
                    print('  Exception="{}"'.format(task.exception()))
            else:
                print('  Status done={} cancelled={}'.format(task.done(), task.cancelled()))
        except Exception:
            #print('Exception:', e)
            #traceback.print_exc()
            traceback.print_exc(file=sys.stdout)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            
    print('Ending...')
    return [t1, t2, t3]


print()

loop = asyncio.get_event_loop()
#loop.set_debug(True)

task = loop.create_task(run())

loop.create_task(stop(loop))
loop.run_forever()
print(task.exception())

