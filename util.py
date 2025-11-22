import time
from contextlib import contextmanager

@contextmanager
def timer(label:str ="Block"):
    """
    A simple context manager to time a block of code using perf_counter.
    """
    # Setup (__enter__)
    start = time.perf_counter()

    # Yield control to the 'with' block
    try:
        yield

    # Cleanup (__exit__) - runs after the 'with' block completes or raises an exception
    finally:
        end = time.perf_counter()
        elapsed = end - start
        # print(f"[{label}] took {elapsed:.4f}s")
