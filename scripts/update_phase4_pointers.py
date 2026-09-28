"""Point the loop state at the current Phase 4 design, power analysis and A/B choice.

Idempotent and atomic (reuses the tested ``publish_many``).  The V1 design draft is
recorded as superseded here and kept byte-identical; nothing is deleted: any legacy
Phase 4 pointer keys move into ``history_as_recorded_before_v2``.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correct_phase3_addendum import _dump, publish_many

STATE = "results/sft_root_cause_state.json"
V1_DRAFT = "docs/SFT_ROOT_CAUSE_PHASE4_DESIGN_DRAFT.md"
CURRENT = {"design": "docs/SFT_ROOT_CAUSE_PHASE4_DESIGN_DRAFT_V2.md",
           "power_analysis": "results/sft_root_cause_phase4_power_analysis_v1.json",
           "choices": "docs/SFT_ROOT_CAUSE_PHASE4_CHOICES.md",
           "census": "results/sft_root_cause_phase4_cohort_census.json"}
EVENT = "phase 4 redesign (V2 loss-mask design, simulation power, A/B choice)"


def sha(root: Path, rel: str) -> str:
    return hashlib.sha256((root / rel).read_bytes()).hexdigest()


def apply(state: dict, root: Path) -> dict:
    state = copy.deepcopy(state)
    phase4 = state["phases"][4]
    legacy = [k for k in ("design_draft", "preflight_census") if k in phase4]
    if legacy:
        phase4.setdefault("history_as_recorded_before_v2", {})
        for key in legacy:
            phase4["history_as_recorded_before_v2"][key] = phase4.pop(key)
    phase4["status"] = "awaiting_user_choice_A_or_B"
    phase4["frozen"] = False
    phase4["training_started"] = False
    phase4["current"] = {name: {"path": rel, "sha256": sha(root, rel)}
                         for name, rel in CURRENT.items()}
    phase4["superseded_designs"] = [{"path": V1_DRAFT, "sha256": sha(root, V1_DRAFT),
                                     "superseded_by": CURRENT["design"],
                                     "reason": "changed input conditioning and loss masking "
                                               "together (not single-factor)"}]
    if not any(entry.get("event") == EVENT for entry in state["log"]):
        state["log"].append({
            "utc": "2026-09-28", "event": EVENT,
            "finding": ("only the label mask differs in the V2 primary design; a five-point "
                        "confirmatory gate needs ~352-500 groups for a true 10-point effect and "
                        "is unreachable for a true 5-point effect; strict untouched pool (33 "
                        "feasible) unusable; repository-disjoint Phase 6 unresolved"),
            "next_decision": "user: Choice A (acquire) or Choice B (internal exploratory pilot); "
                             "no GPU step without separate authorisation"})
    return state


def run(root: Path = ROOT) -> str:
    state = json.loads((root / STATE).read_text(encoding="utf-8"))
    updated = apply(state, root)
    for name, block in updated["phases"][4]["current"].items():
        if sha(root, block["path"]) != block["sha256"]:
            raise SystemExit(f"REFUSED: {name} pointer does not resolve")
    data = _dump(updated)
    if (root / STATE).read_bytes() != data:
        publish_many(root, {STATE: data})
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    print(run())
