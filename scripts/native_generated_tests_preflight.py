"""Preflight for the frozen native generated-test protocol v1 (CPU only; launches nothing).

Checks what exists and names every blocker. Readiness is reported separately for the dress
rehearsal (24 development targets) and for confirmation (a repository-disjoint cohort).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

OUTPUT = "results/sft_root_cause_native_generated_tests_preflight_v1.json"
PROTOCOL = "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V1.md"
PARENT = "docs/REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md"
BRANCH_RECEIPT = "results/sft_root_cause_phase4_receiver_capture_receipt_v2.json"
DEV_MANIFEST = "results/sft_root_cause_phase4_receiver_capture_manifest_v2.json"
MODEL, REVISION = "Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a"
ADAPTER = "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter"
ADAPTER_SHA256 = "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"
COMPONENTS = {
    "permitted_view_prompt_builder": "harness/native_generated_test_prompt.py",
    "leakage_scanner": "harness/native_generated_test_leakage.py",
    "durable_generation_runner": "scripts/native_generated_tests_generate.py",
    "native_injection_executor": "scripts/native_generated_tests_execute_wsl.py",
    "repository_atheris_adapter": "scripts/native_generated_tests_atheris_wsl.py",
    "frozen_analysis": "scripts/native_generated_tests_analyse.py",
}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    return "/mnt/" + resolved.drive[0].lower() + resolved.as_posix()[2:]


def atheris_status() -> dict:
    """Runs a tracked helper script (inline ``bash -c`` strings are mangled by wsl.exe)."""
    try:
        done = subprocess.run(["wsl.exe", "-u", "root", "--", "bash",
                               wsl_path(ROOT / "scripts" / "wsl_atheris_check.sh")],
                              capture_output=True, text=True, timeout=120)
        out = done.stdout.strip().split()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "detail": type(exc).__name__}
    ok = len(out) == 2 and out[0] == "2.3.0"
    return {"available": ok, "atheris": out[0] if out else None,
            "python": out[1] if len(out) > 1 else None}


def main() -> int:
    from harness.acquisition_receipt import ProtectedAccessMonitor
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    branch = load(BRANCH_RECEIPT)
    dev = load(DEV_MANIFEST)
    blockers, notes = [], []
    if branch["branch"] != "B":
        blockers.append("the receiver-aware pilot did not select Branch B")
    try:
        import transformers
        transformers.AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        base_local = True
    except Exception as exc:          # recorded, not hidden
        base_local = False
        blockers.append(f"base model snapshot not local: {type(exc).__name__}")
    adapter_file = ROOT / ADAPTER / "adapter_model.safetensors"
    adapter_ok = adapter_file.exists() and sha(f"{ADAPTER}/adapter_model.safetensors") == \
        ADAPTER_SHA256
    if not adapter_ok:
        blockers.append("frozen SFT adapter missing or hash differs")
    atheris = atheris_status()
    if not atheris["available"]:
        blockers.append("atheris 2.3.0 on WSL Python 3.11 unavailable")
    components = {name: (ROOT / rel).exists() for name, rel in COMPONENTS.items()}
    blockers += [f"component not implemented: {name} ({COMPONENTS[name]})"
                 for name, present in components.items() if not present]
    repos = sorted({t["repository"] for t in dev["targets"]})
    rehearsal = {"targets": len(dev["targets"]), "repositories": len(repos),
                 "meets_parent_rehearsal_pool": 20 <= len(dev["targets"]) <= 30
                 and len(repos) >= 5,
                 "status": "development targets; pipeline shake-out only; never confirmation"}
    confirmation_blockers = [
        "no repository-disjoint isolation-v6-admitted confirmation cohort exists",
        "acquiring one needs separate approval (mass acquisition is refused; the fresh "
        "A-prime pilot admitted 5/120 candidates)"]
    evidence = ProtectedAccessMonitor.evidence(mark)
    if evidence["protected_paths_opened"]:
        blockers.append(f"protected paths opened {evidence['protected_paths_opened']}")
    receipt = {
        "schema_version": "oneiros_native_generated_tests_preflight_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "protocol": {"path": PROTOCOL, "sha256": sha(PROTOCOL)},
        "parent_protocol": {"path": PARENT, "sha256": sha(PARENT)},
        "branch_receipt": {"path": BRANCH_RECEIPT, "sha256": sha(BRANCH_RECEIPT)},
        "models": {"base": {"name": MODEL, "revision": REVISION, "local": base_local},
                   "sft": {"adapter": ADAPTER, "sha256": ADAPTER_SHA256,
                           "verified": adapter_ok}},
        "atheris": atheris, "components": components,
        "dress_rehearsal_cohort": {**rehearsal, "manifest": {"path": DEV_MANIFEST,
                                                             "sha256": sha(DEV_MANIFEST)}},
        "readiness": {"dress_rehearsal_ready": not blockers,
                      "confirmation_ready": False,
                      "blockers": blockers, "confirmation_blockers": confirmation_blockers},
        "gpu_generation_launched": False, "protected_access_audit":
            {k: v for k, v in evidence.items() if k != "opens_checked"},
        "root_cause_established": False, "generalization_established": False,
        "next_permitted_action": ("implement the named components (CPU) and re-run this "
                                  "preflight; dress-rehearsal GPU generation then needs "
                                  "explicit approval; confirmation needs an approved "
                                  "repository-disjoint cohort"),
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps(receipt["readiness"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
