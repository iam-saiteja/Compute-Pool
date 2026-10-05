"""Execution strategies for Compute Pool.

The paradigms the platform must support -- independent/map, intra-node
multi-GPU, periodic-sync data parallel, cross-node pipeline, inference serving,
and (eventually) tight synchronous training -- differ mainly in how much and
how tightly they communicate between workers. The fabric has a measured
ceiling (bandwidth, latency, no direct peer-to-peer, no inbound), so each
strategy declares its communication profile and whether it is feasible on a
given fabric. The scheduler filters to feasible strategies and fails safely
when a job needs more than the fabric can give, instead of pretending
independent workers are one cluster.

This module is the contract and the registry. A strategy's actual run loop
lives in its own module under strategies/ and is launched on the workers; the
scheduler only needs the descriptor and the feasibility check.
"""
from dataclasses import dataclass, field


@dataclass
class FabricProfile:
    """What the transport between two workers can actually do (measured)."""
    bandwidth_mbps: float          # effective, both directions combined
    rtt_ms: float                  # round trip for a small message
    direct_p2p: bool               # can two workers open a direct connection to each other
    inbound: bool                  # can a worker accept an inbound connection


# The current Kaggle-over-SSH fabric, from examples/cluster_comm/link_bench.py.
KAGGLE_SSH_FABRIC = FabricProfile(bandwidth_mbps=16.0, rtt_ms=70.0, direct_p2p=False, inbound=False)


@dataclass
class CommProfile:
    """How a strategy communicates between nodes."""
    # 'none' | 'periodic' | 'per_step' | 'per_microbatch'
    cross_node_frequency: str
    # bytes exchanged with the rest of the cluster per unit above (rough)
    cross_node_bytes: int = 0
    needs_direct_p2p: bool = False
    tolerates_worker_loss: bool = True


@dataclass
class Strategy:
    key: str
    summary: str
    comm: CommProfile
    # minimum fraction of an fp16-sized step we are willing to spend on the wire
    # before the strategy is considered starved on a given fabric
    max_wire_fraction: float = 0.9

    def feasible(self, fabric: FabricProfile):
        """Return (ok, reason). Fail safely: a clear reason beats a broken run."""
        if self.comm.needs_direct_p2p and not fabric.direct_p2p:
            return False, "needs direct worker-to-worker connections; this fabric has none"
        if self.comm.cross_node_frequency == "none":
            return True, "no cross-node communication"
        # crude starvation check: transfer time vs a nominal 0.2s/step of compute
        if self.comm.cross_node_bytes:
            transfer_s = self.comm.cross_node_bytes / (fabric.bandwidth_mbps * 1024 * 1024)
            if self.comm.cross_node_frequency in ("per_step", "per_microbatch"):
                if transfer_s > self.max_wire_fraction:
                    return False, (f"~{transfer_s:.1f}s on the wire per step at "
                                   f"{fabric.bandwidth_mbps:.0f} MB/s would starve compute")
        return True, "within fabric limits"


_REGISTRY: dict[str, Strategy] = {}


def register(strategy: Strategy):
    _REGISTRY[strategy.key] = strategy
    return strategy


def get(key: str) -> Strategy:
    if key not in _REGISTRY:
        raise KeyError(f"unknown strategy {key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def all_strategies():
    return dict(_REGISTRY)


def feasible_strategies(fabric: FabricProfile):
    return {k: s for k, s in _REGISTRY.items() if s.feasible(fabric)[0]}


# The strategies the platform ships. Run loops live under strategies/.
register(Strategy(
    "independent",
    "One task per GPU, no cross-node communication (map, sweeps, batch inference).",
    CommProfile(cross_node_frequency="none"),
))
register(Strategy(
    "intra_node_ddp",
    "Multi-GPU within a single worker via local NCCL; no cross-node communication.",
    CommProfile(cross_node_frequency="none"),
))
register(Strategy(
    "data_parallel_sync",
    "Data parallel with periodic parameter sync over the link (small models, LoRA adapters).",
    CommProfile(cross_node_frequency="periodic", cross_node_bytes=4 * 1024 * 1024),
))
register(Strategy(
    "pipeline",
    "Model split into stages across nodes; activations and gradients cross the link each micro-batch.",
    CommProfile(cross_node_frequency="per_microbatch", cross_node_bytes=4 * 1024 * 1024),
))
register(Strategy(
    "sync_collective",
    "Tight synchronous cross-node collectives (full all-reduce every step).",
    CommProfile(cross_node_frequency="per_step", cross_node_bytes=512 * 1024 * 1024,
                needs_direct_p2p=True, tolerates_worker_loss=False),
))
