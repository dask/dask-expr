from __future__ import annotations

import pytest

distributed = pytest.importorskip("distributed")

import asyncio
import os

from dask.distributed import Worker
from distributed.scheduler import Scheduler
from distributed.utils import Deadline
from distributed.utils_test import client as c  # noqa F401
from distributed.utils_test import gen_cluster

import dask_expr as dx


async def clean_worker(
    worker: Worker, interval: float = 0.01, timeout: int | None = None
) -> None:
    """Assert that the worker has no shuffle state"""
    deadline = Deadline.after(timeout)
    extension = worker.extensions["shuffle"]

    while extension._runs and not deadline.expired:
        await asyncio.sleep(interval)
    for dirpath, dirnames, filenames in os.walk(worker.local_directory):
        assert "shuffle" not in dirpath
        for fn in dirnames + filenames:
            assert "shuffle" not in fn


async def clean_scheduler(
    scheduler: Scheduler, interval: float = 0.01, timeout: int | None = None
) -> None:
    """Assert that the scheduler has no shuffle state"""
    deadline = Deadline.after(timeout)
    extension = scheduler.extensions["shuffle"]
    while extension.states and not deadline.expired:
        await asyncio.sleep(interval)
    assert not extension.states
    assert not extension.heartbeats


@pytest.mark.parametrize("npartitions", [None, 1, 20])
@gen_cluster(client=True)
async def test_p2p_shuffle(c, s, a, b, npartitions):
    df = dx.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = df.shuffle("x", backend="p2p", npartitions=npartitions)
    if npartitions is None:
        assert out.npartitions == df.npartitions
    else:
        assert out.npartitions == npartitions
    x, y, z = c.compute([df.x.size, out.x.size, out.partitions[-1].x.size])
    x = await x
    y = await y
    z = await z
    assert x == y
    if npartitions != 1:
        assert x > z

    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)
