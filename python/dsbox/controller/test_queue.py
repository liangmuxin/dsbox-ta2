import asyncio
import functools
import os
import sys
import time

path = os.path.abspath('../..')
print('Add to python path: ', path)
sys.path.append(path)

from importlib import reload

import dsbox.controller.dsbox_runtime
reload(dsbox.controller.dsbox_runtime)
from dsbox.controller.dsbox_runtime import ProcessExecutorQueue

def run_pipeline(id, timeout=5):
    print('run_pipeline {} start'.format(id))
    time.sleep(timeout)
    print('run_pipeline {} done'.format(id))
    return ('pipeline {} done'.format(id))

loop = asyncio.get_event_loop()
async def run():
    executor_queue = ProcessExecutorQueue(loop)
    t1 = executor_queue.run(functools.partial(run_pipeline, 1, timeout=3))
    t2 = executor_queue.run(functools.partial(run_pipeline, 2, timeout=2))
    t3 = executor_queue.run(functools.partial(run_pipeline, 3, timeout=1))

    # give the task some time to run
    done, pending = await asyncio.wait([t1, t2, t3], timeout=1)
    print('done, pending', len(done), len(pending))
    print('# running tasks', executor_queue.running_task_size())
    print('# done    tasks', executor_queue.done_task_size())
    while executor_queue.all_task_size() > 0:
        result = await executor_queue.get()
        print('# running tasks', executor_queue.running_task_size())
        print('# done    tasks', executor_queue.done_task_size())
        print('got_task: ', type(result), result, result)
        await asyncio.sleep(1)
    print('run done')

task = loop.create_task(run())
loop.run_forever()
#loop.run_until_complete(asyncio.wait([task]))
loop.close()

