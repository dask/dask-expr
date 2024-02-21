from collections import defaultdict
from typing import ClassVar

from crick import TDigest
from distributed import Scheduler, SchedulerPlugin, Worker, WorkerPlugin
from distributed.protocol.pickle import dumps


class AnalyzePlugin(SchedulerPlugin):
    idempotent: ClassVar[bool] = True
    name: ClassVar[str] = "analyze"
    _scheduler: Scheduler | None

    def __init__(self) -> None:
        self._scheduler = None

    async def start(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler
        scheduler.handlers["analyze_get_statistics"] = self.get_statistics
        worker_plugin = _AnalyzeWorkerPlugin()
        await self._scheduler.register_worker_plugin(
            None,
            dumps(worker_plugin),
            name=worker_plugin.name,
        )

    async def get_statistics(self, id: str):
        assert self._scheduler is not None
        worker_statistics = await self._scheduler.broadcast(
            msg={"op": "analyze_get_statistics", "id": id}
        )
        cluster_statistics = defaultdict(TDigest)
        for statistics in worker_statistics.values():
            for group_key, digest in statistics.items():
                cluster_statistics[group_key].merge(digest)
        return dict(cluster_statistics)


class _AnalyzeWorkerPlugin(WorkerPlugin):
    idempotent: ClassVar[bool] = True
    name: ClassVar[str] = "analyze"
    _digests: defaultdict[str, defaultdict[str, TDigest]]
    _worker: Worker | None

    def __init__(self) -> None:
        self._worker = None

    def setup(self, worker: Worker) -> None:
        self._digests = defaultdict(lambda: defaultdict(TDigest))
        self._worker = worker
        self._worker.handlers["analyze_get_statistics"] = self.get_statistics

    def add(self, id: str, expr_name: str, nbytes: int):
        self._digests[id][expr_name].add(nbytes)

    def get_statistics(self, id: str) -> dict[str, TDigest]:
        return dict(self._digests[id])


def get_worker_plugin() -> _AnalyzeWorkerPlugin:
    from distributed import get_worker

    try:
        worker = get_worker()
    except ValueError as e:
        raise RuntimeError(
            "``.analyze()`` requires Dask's distributed scheduler"
        ) from e

    try:
        return worker.plugins["analyze"]  # type: ignore
    except KeyError as e:
        raise RuntimeError(
            f"The worker {worker.address} does not have an Analyze plugin."
        ) from e
