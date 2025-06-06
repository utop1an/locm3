from multiprocessing.pool import ThreadPool
from typing import Callable, Union


def set_timer_throw_exc(
    num_seconds: Union[float, int, None],
    exception: Callable,
    *exception_args,
    **exception_kwargs,
):
    def timer(function):
        """
        Checks that a function runs within the specified time and raises an exception if it doesn't.

        Args:
            function (function reference):
                The generator function to be wrapped with this time-checker.

        Returns:
            The wrapped function.
        """

        def wrapper(*args, **kwargs):
            # If num_seconds is None, run the function without a timeout
            if num_seconds is None:
                return function(*args, **kwargs)
            
            pool = ThreadPool(processes=1)

            thr = pool.apply_async(function, args=args, kwds=kwargs)
            # run the function for the specified seconds
            thr.wait(num_seconds)
            # return successful results, if ready
            if thr.ready():
                pool.terminate()
                return thr.get()
            else:
                # otherwise, raise an exception if the function takes too long
                raise exception(*exception_args, **exception_kwargs)

        return wrapper

    return timer

def basic_timer(num_seconds: Union[float, int]):
    def timer(function):
        """
        Runs a function for a specified time.

        Returns:
            The wrapped function.
        """

        def wrapper(*args, **kwargs):
            pool = ThreadPool(processes=1)

            thr = pool.apply_async(function, args=args, kwds=kwargs)
            # run the function for the specified seconds and exit without checking for/returning results
            thr.wait(num_seconds)
            pool.terminate()
            return

        return wrapper

    return timer