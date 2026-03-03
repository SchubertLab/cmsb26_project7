# from AdVist lecture Pythonics 2
# modified with chattie

import time
import functools


class TimeWrapper:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)
    
    def __get__(self, instance, owner):
        # Bind the function to the instance
        return lambda *args, **kwargs: self(instance, *args, **kwargs)
    
    def __call__(self, *args, **kwds):
        start = time.time()
        result = self.func(*args, **kwds)
        end = time.time()
        runtime = end - start

        # format the argument passed to the function
        args_s = ', '.join([str(x) for x in args])
        kwds_s = ', '.join([f'{k}={v}' for k, v in kwds.items()])
        args_kwds_s = ', '.join(filter(None, [args_s, kwds_s]))

        # print the collected info
        print(self.func.__name__, round(runtime, 0), ' sec.') # f'({args_kwds_s}):', 

        return result
