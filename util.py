import time
from contextlib import contextmanager
import os
import datetime

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


def parse_manifest(manifest: str):
    return [f.strip() for f in open(manifest, "r").readlines()]


def ensure_dir_exists(d):
    if d is None or d == "":
        return
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except FileExistsError:
            # can happen as race condition
            pass


def ensure_dir_exists_for_file(f):
    ensure_dir_exists(os.path.dirname(f))


def DTS():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
