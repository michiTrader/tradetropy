"""
Adapter for running evaluations in parallel using a process pool.

The worker (_worker_fn) lives at module level to be picklable with
the 'spawn' start method (Windows / macOS). It is initialized once
per child process via _init_pool, avoiding data reload on every call.
"""

import multiprocessing as mp
import platform
from functools import partial
from typing import Any, Callable, List

from tradetropy.optimize.task import Candidate, Result

# -- Worker global variables ---------------------------------------------------
# Only used inside child processes after _init_pool().
_worker_evaluate_fn: Callable | None = None
_worker_data: Any = None
# Opened shared_memory handles for the current worker; kept alive for the
# worker's whole lifetime so the mapped array views stay valid.
_worker_shm: list = []


def _init_pool(evaluate_fn: Callable, data: Any) -> None:
    """
    Pool initializer: saves evaluate_fn and data once per child process.
    Passed as `initializer` to mp.Pool.

    When ``data`` is a shared-memory bundle (see _shm_bundle.build_shm_bundle),
    the large input matrices travel in shared_memory instead of the pickle
    stream: it is hydrated once here into the normal data bundle, mapping each
    block zero-copy into this worker.
    """
    global _worker_evaluate_fn, _worker_data
    _worker_evaluate_fn = evaluate_fn
    if isinstance(data, dict) and data.get("_shm_bundle"):
        from tradetropy.backtest._shm_bundle import hydrate_bundle
        data = hydrate_bundle(data, _worker_shm)
    _worker_data = data


def _worker_fn(candidate: Candidate) -> Result:
    """
    Function executed in each child process.
    Receives a single Candidate and returns a Result.
    Depends on global variables initialized by _init_pool.
    """
    return _worker_evaluate_fn(candidate, _worker_data)


class PoolEvaluator:
    """
    Encapsulates parallel candidate execution using a process pool.

    Parameters
    ----------
    evaluate_fn : callable(Candidate, data) -> Result
        Evaluation function. Must be defined at module level
        (not a lambda or closure) to be picklable with spawn.
    data        : any serializable object that evaluate_fn needs.
    workers     : number of parallel processes. None -> cpu_count().
    """

    def __init__(
        self,
        evaluate_fn: Callable[[Candidate, Any], Result],
        data: Any,
        workers: int | None = None,
        progress: bool = True,
        desc: str = "Optimize",
        chunksize: int = 1,
    ):
        self.evaluate_fn = evaluate_fn
        self.data = data
        self.workers = workers or mp.cpu_count()
        self.progress = progress
        self.desc = desc
        self.chunksize = max(1, int(chunksize))

    def evaluate(self, candidates: List[Candidate]) -> List[Result]:
        """
        Evaluates a list of candidates in parallel and returns results
        in the same order as the input list.

        A single aggregate progress bar (one for the whole optimization, not
        one per backtest) counts candidates as they finish. ``imap`` yields
        each result as soon as it is ready while preserving input order, so the
        bar advances per completed backtest; ``chunksize`` only batches task
        dispatch to workers and does not change per-backtest engine speed.
        """
        if not candidates:
            return []

        ctx_name = "fork" if platform.system() == "Linux" else "spawn"
        ctx = mp.get_context(ctx_name)

        with ctx.Pool(
            processes=self.workers,
            initializer=_init_pool,
            initargs=(self.evaluate_fn, self.data),
        ) as pool:
            result_iter = pool.imap(
                _worker_fn, candidates, chunksize=self.chunksize
            )
            result_iter = self._wrap_progress(result_iter, len(candidates))
            results = list(result_iter)

        return results

    def _wrap_progress(self, iterable, total: int):
        """Wrap the result iterator in a single tqdm bar (best-effort)."""
        if not self.progress:
            return iterable
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            return iterable
        return _tqdm(
            iterable, total=total, desc=self.desc, unit="bt",
            dynamic_ncols=True, leave=True,
        )