import os
import sys

def setup_determinism_env(single_threaded_cpu: bool = True, gpu_deterministic: bool = True) -> None:
    """Configure environment variables for hardware determinism before JAX/BLAS runtimes initialize.

    This locks down thread counts and compiler flags so that floating-point reductions and
    operations produce reproducible results across heterogeneous hardware.
    """
    if single_threaded_cpu:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    flags = []
    if gpu_deterministic:
        flags.extend(["--xla_gpu_deterministic_ops=true", "--xla_gpu_ftz=false"])
    if single_threaded_cpu:
        flags.extend(["--xla_cpu_multi_thread_eigen=false", "intra_op_parallelism_threads=1"])

    current_xla_flags = os.environ.get("XLA_FLAGS", "")
    for flag in flags:
        if flag not in current_xla_flags:
            current_xla_flags = f"{current_xla_flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = current_xla_flags
