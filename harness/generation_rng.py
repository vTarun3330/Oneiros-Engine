"""One place that seeds generation, for every path that generates.

The frozen receipt recorded ``generation_seed: 42`` and the sealed path never
applied it. Locked validation did - ``torch.manual_seed`` and
``torch.cuda.manual_seed_all`` immediately before generation - but that call
lived inline in the evaluator, so the sealed path inherited the *claim* of a
seed without the act of seeding.

A seed recorded but not applied is worse than no seed at all: the artifact
asserts reproducibility that does not exist, and nobody can tell from the
artifact alone. So the seeding is lifted here, both paths call it, and the run
artifact records that the call happened rather than that a number was written
down.

``SEED_APPLICATION_VERSION`` names which RNGs are touched and in what order, so
a later change to that set is visible as a version change rather than as a
silently different stream.
"""
from __future__ import annotations

import random
from typing import Any, Dict

#: Bump when the set of seeded RNGs, or the order, changes. Recorded beside
#: every seeded run so "seed 42" is never ambiguous about what it seeded.
SEED_APPLICATION_VERSION = "oneiros_generation_rng_v1"

#: What this initializer touches, in order. Matches what locked validation did.
SEEDED_GENERATORS = ("python_random", "torch_cpu", "torch_cuda_all")


def seed_generation_rngs(seed: int) -> Dict[str, Any]:
    """Seed every RNG that can influence generation, and say so.

    Returns a record for the run artifact. The record is evidence that the call
    happened; it deliberately reports the CUDA device count actually seeded,
    because "seeded CUDA" on a machine with no CUDA is a different claim.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"generation seed must be a non-negative int, got {seed!r}")

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    cuda_devices = 0
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_devices = torch.cuda.device_count()

    return {
        "seed_application_version": SEED_APPLICATION_VERSION,
        "seed": seed,
        "seeded": list(SEEDED_GENERATORS),
        "cuda_devices_seeded": cuda_devices,
        "applied": True,
    }


def rng_state_fingerprint() -> Dict[str, str]:
    """A cheap fingerprint of current RNG state, for before/after evidence.

    Not a reproducibility guarantee on its own - it exists so a run artifact can
    show that the state actually moved when the reset was applied, rather than
    asserting a reset that no-oped.
    """
    import hashlib

    import torch

    python_state = repr(random.getstate()).encode("utf-8")
    torch_state = torch.get_rng_state().numpy().tobytes()
    return {
        "python_random_sha256": hashlib.sha256(python_state).hexdigest(),
        "torch_cpu_sha256": hashlib.sha256(torch_state).hexdigest(),
    }
