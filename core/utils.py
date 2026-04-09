import numpy as np


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