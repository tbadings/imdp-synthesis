import logging
from collections import deque

import numpy as np


class _CleanConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f'ERROR: {message}'
        if record.levelno >= logging.WARNING:
            return f'WARN: {message}'
        if record.levelno == logging.DEBUG:
            return f'DEBUG: {message}'
        return message


def configure_logging(log_level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_CleanConsoleFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def add_file_handler(log_dir, stamp: str = '') -> None:
    """Add a file handler that writes all log output to run_<stamp>.log in log_dir."""
    filename = f'run_{stamp}.log' if stamp else 'run.log'
    file_handler = logging.FileHandler(log_dir / filename, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S',
    ))
    logging.getLogger().addHandler(file_handler)


def create_batches(data_length, batch_size):
    '''
    Create batches for the given data and batch size. Returns the start and end indices to iterate over.

    :param data_length: Total number of data points.
    :param batch_size: Number of points per batch.
    :return: Each batch is represented by the slice [starts[i]:ends[i]].
    '''

    num_batches = np.ceil(data_length / batch_size).astype(int)
    starts = (np.arange(num_batches) * batch_size).astype(int)
    ends = (np.minimum(starts + batch_size, data_length)).astype(int)

    return starts, ends


def lexsort4d(array):
    idxs = np.lexsort((
        array[:, 3],
        array[:, 2],
        array[:, 1],
        array[:, 0]
    ))

    return array[idxs]


def cm2inch(*tupl):
    '''
    Convert centimeters to inches
    '''

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def remove_consecutive_duplicates(trace):
    '''
    Remove consecutive duplicates from a given trace.

    :param trace:
    :return: Trace without duplicates
    '''
    done = False
    i = 0
    while not done:
        # If same as next entry, remove it
        if i >= len(trace) - 1:
            done = True
        else:
            if np.all(trace[i] == trace[i + 1]):
                trace = np.delete(trace, i + 1, axis=0)
            else:
                i += 1

    return trace


def jit_compile_count(jitted_function) -> int | None:
    """Return JIT cache size if available, else ``None``."""
    cache_size_fn = getattr(jitted_function, "_cache_size", None)
    if cache_size_fn is None:
        return None
    try:
        return int(cache_size_fn())
    except Exception:
        return None


class DPStagnationCheck:
    """
    Early termination of dynamic programming when the tracked value stagnates.

    Terminates once the tracked value (the minimum value over the initial states) has improved by
    less than ``improvement`` over the last ``iterations`` iterations. Termination is never triggered
    before ``min_iterations`` full iterations have been performed (the window keeps sliding in the
    meantime). Disabled (never terminates) if either ``iterations`` or ``improvement`` is zero, so that
    dynamic programming runs until convergence.
    """

    def __init__(self, iterations: int, improvement: float, min_iterations: int = 0):
        self.iterations = int(iterations)
        self.improvement = float(improvement)
        self.min_iterations = int(min_iterations)
        self.enabled = self.iterations > 0 and self.improvement > 0
        self.history = deque(maxlen=self.iterations + 1)
        self.num_updates = 0

    @property
    def iterations_done(self) -> int:
        """Number of full iterations recorded (the first update is the value before the first iteration)."""
        return max(self.num_updates - 1, 0)

    def update(self, value: float) -> bool:
        """
        Record the tracked value (call once before the first iteration and once after every iteration).

        :param value: Current tracked value
        :return: True if dynamic programming should terminate
        """
        if not self.enabled:
            return False
        self.history.append(float(value))
        self.num_updates += 1
        return (self.iterations_done >= self.min_iterations
                and len(self.history) == self.history.maxlen
                and self.gain < self.improvement)

    @property
    def gain(self) -> float:
        """Improvement of the tracked value over the current window."""
        return self.history[-1] - self.history[0]

    def describe(self) -> str:
        desc = (f'stop if min v[s0] improves by less than {self.improvement:g} '
                f'over {self.iterations} iterations')
        if self.min_iterations > 0:
            desc += f' (after at least {self.min_iterations} iterations)'
        return desc

    def message(self, iteration: int) -> str:
        return (f'Early termination after {iteration} iterations: min v[s0]={self.history[-1]:.6f} improved by '
                f'{self.gain:.6f} < {self.improvement:g} over the last {self.iterations} iterations')
