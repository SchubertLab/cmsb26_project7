# from AdVist lecture Pythonics 2, modified a bit

import time
import functools

# helper class printing the runtime of a function
class TimeWrapper:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)
    
    def __get__(self, instance, owner):
        return lambda *args, **kwargs: self(instance, *args, **kwargs)
    
    def __call__(self, *args, **kwds):
        # run the function and determine its runtime
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
