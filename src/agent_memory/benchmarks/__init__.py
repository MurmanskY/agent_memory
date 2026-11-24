from __future__ import annotations

from pathlib import Path

from .base import Benchmark
from .locomoco import LoCoMoBenchmark
from .longmemeval import LongMemEvalBenchmark
from .memoryagentbench import MemoryAgentBenchBenchmark
from .gatemem import GateMemBenchmark

BENCHMARKS = {
    "locomoco": LoCoMoBenchmark,
    "longmemeval": LongMemEvalBenchmark,
    "memoryagentbench": MemoryAgentBenchBenchmark,
    "gatemem": GateMemBenchmark,
}


def get_benchmark(name: str, data_path: str | Path, **kwargs) -> Benchmark:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {name} (have: {sorted(BENCHMARKS)})")
    return BENCHMARKS[name](data_path, **kwargs) if kwargs else BENCHMARKS[name](data_path)


__all__ = [
    "BENCHMARKS",
    "get_benchmark",
    "Benchmark",
    "LoCoMoBenchmark",
    "LongMemEvalBenchmark",
    "MemoryAgentBenchBenchmark",
    "GateMemBenchmark",
]
