"""Authoritative generation cohort, telemetry schema and per-arm generation loading
(amendment v2.3 sections A-D). Standard library only: imported by the generator (Windows),
the sandbox executor (WSL, CPython 3.13) and the analysis.

* ``resolve_cohort`` loads the exact condition from the exact job artifact and the successor
  manifest; it verifies the job file hash, ``job_sha256``, unique keys, generation targets as a
  subset of the qualified targets, and that the excluded set equals the recorded
  pre-generation exclusions. The generation cohort is NEVER inferred from kept/qualified
  targets.
* ``row_problems`` checks one generation row against telemetry schema
  ``oneiros_native_generation_telemetry_v2`` (amendment v2.4 B): seed in {42, 43, 44},
  key == target_key::seed, target_seed recomputed exactly, 0 < prompt_tokens <= 2048, the
  job's prompt hash, exact candidate/batch counts, per-row CUDA peaks plus separately named
  process-lifetime peaks; exact token IDs (the count includes the first generated EOS and
  excludes later batch padding); ``hit_completion_limit`` only when no EOS was generated and
  exactly max_new_tokens were produced.
* ``load_arm_generations`` loads base/ and sft/ arm directories (or explicit per-arm paths),
  hashing every file and contract and refusing swapped arms, mismatched identities, and
  missing, duplicated, extra, stale or malformed rows.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

COHORT_VERSION = "oneiros_native_generation_cohort_v1"
TELEMETRY_SCHEMA = "oneiros_native_generation_telemetry_v2"
ARMS = ("base", "sft")
SEEDS = (42, 43, 44)
CANDIDATES = 8
BATCH_SIZE = 2
MAX_NEW_TOKENS = 1024
PROMPT_TOKEN_LIMIT = 2048
FINISH_REASONS = ("eos", "length")
# identity fields that legitimately differ between the two arms of one experiment
ARM_SPECIFIC = ("arm", "adapter_manifest_sha256", "prior_arm_verification")


class CohortRefused(SystemExit):
    """The job, manifest or generations are inconsistent; nothing may run."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def contract_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def target_seed(seed: int, target_key: str) -> int:
    """The per-target sampling seed, identical for both arms."""
    return int(hashlib.sha256(f"{seed}:{target_key}".encode()).hexdigest()[:8], 16)


def extract(raw: str) -> Dict[str, Any]:
    """Whole output; strip only a fence that wraps the ENTIRE output."""
    text = raw.strip("\n")
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```" and \
            sum(l.startswith("```") for l in lines) == 2:
        return {"module": "\n".join(lines[1:-1]) + "\n", "fence_stripped": True}
    return {"module": raw if raw.endswith("\n") else raw + "\n", "fence_stripped": False}


# --- cohort -----------------------------------------------------------------------------------

def cohort_fields(qualified: List[Mapping[str, Any]], job_rel: str, job_bytes: bytes,
                  condition: str, exclusion_detail: Mapping[str, Mapping[str, Any]]
                  ) -> Dict[str, Any]:
    """Manifest fields that make the cohort explicit (used by the v2.3 rebuild and by the
    synthetic pipeline). ``qualified``: target dicts with ``key`` and ``repository``."""
    job = json.loads(job_bytes.decode("utf-8"))[condition]
    generation = sorted(i["target_key"] for i in job["items"])
    refused = {r["target_key"]: r for r in job["refused"]}
    qualified_keys = [t["key"] for t in qualified]
    exclusions = []
    for key in qualified_keys:
        if key in generation:
            continue
        if key not in refused:
            raise CohortRefused(f"REFUSED: qualified target {key} neither generated nor refused")
        exclusions.append({"target_key": key, "stage": "pre_generation",
                           "reasons": list(refused[key]["reasons"]),
                           **dict(exclusion_detail.get(key, {}))})
    return {"cohort_version": COHORT_VERSION,
            "qualified_targets": qualified_keys, "generation_targets": generation,
            "pre_generation_exclusions": exclusions,
            "job": {"path": job_rel, "file_sha256": sha256_bytes(job_bytes),
                    "job_sha256": job["job_sha256"], "condition": condition},
            "expected": expected_sizes(len(generation))}


def expected_sizes(n_targets: int) -> Dict[str, int]:
    return {"generation_targets": n_targets, "rows_per_arm": n_targets * len(SEEDS),
            "candidates_per_row": CANDIDATES,
            "candidates_per_arm": n_targets * len(SEEDS) * CANDIDATES,
            "candidates_total": n_targets * len(SEEDS) * CANDIDATES * len(ARMS)}


def resolve_cohort(job_path: Path, manifest_path: Path,
                   condition: str = "primary_whole_module") -> Dict[str, Any]:
    job_bytes = Path(job_path).read_bytes()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    problems = []
    for field in ("qualified_targets", "generation_targets", "pre_generation_exclusions", "job"):
        if field not in manifest:
            raise CohortRefused(f"REFUSED: manifest has no explicit {field!r}; the generation "
                                "cohort is never inferred from kept targets")
    bound = manifest["job"]
    if sha256_bytes(job_bytes) != bound.get("file_sha256"):
        problems.append("job file hash differs from the manifest")
    if bound.get("condition") != condition:
        problems.append("condition differs from the manifest")
    job_file = json.loads(job_bytes.decode("utf-8"))
    if condition not in job_file:
        raise CohortRefused(f"REFUSED: condition {condition!r} not in the job artifact")
    job = job_file[condition]
    if contract_sha({"items": job["items"]}) != job.get("job_sha256") or \
            job.get("job_sha256") != bound.get("job_sha256"):
        problems.append("job_sha256 mismatch")
    keys = [i["target_key"] for i in job["items"]]
    qualified = list(manifest["qualified_targets"])
    excluded = {e["target_key"] for e in manifest["pre_generation_exclusions"]}
    if len(keys) != len(set(keys)) or len(qualified) != len(set(qualified)):
        problems.append("duplicate target keys")
    if sorted(keys) != sorted(manifest["generation_targets"]):
        problems.append("job items differ from the manifest generation targets")
    if not set(keys) <= set(qualified):
        problems.append("generation targets are not a subset of the qualified targets")
    if set(qualified) - set(keys) != excluded:
        problems.append("excluded set differs from the recorded pre-generation exclusions")
    if any(e.get("stage") != "pre_generation" for e in manifest["pre_generation_exclusions"]):
        problems.append("an exclusion is not a pre-generation exclusion")
    if {r["target_key"] for r in job["refused"]} != excluded:
        problems.append("job refusals differ from the recorded exclusions")
    repo_of = {t["key"]: t["repository"] for t in manifest.get("targets", [])}
    if any(k not in repo_of for k in qualified):
        problems.append("repository mapping missing for a qualified target")
    if problems:
        raise CohortRefused(f"REFUSED: generation cohort inconsistent: {problems}")
    return {"cohort_version": COHORT_VERSION, "condition": condition,
            "nature": manifest.get("nature"), "study_mode": manifest.get("study_mode"),
            "requalification_records": manifest.get("requalification_records"),
            "qualified": sorted(qualified), "generation": sorted(keys),
            "pre_generation_exclusions": list(manifest["pre_generation_exclusions"]),
            "repo_of": repo_of, "items": {i["target_key"]: i for i in job["items"]},
            "job_file_sha256": sha256_bytes(job_bytes), "job_sha256": job["job_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "expected": expected_sizes(len(keys))}


# --- preparation binding (amendment v2.4 section C) ----------------------------------------------

PREP_REQUIRED = ("key", "category", "module", "qualname", "python_path", "env_dir", "interpreter",
                 "views", "view_manifest_sha256", "module_sha256", "attestation",
                 "environment_lock")


def resolve_prep(prep_path: Path, manifest_path: Path, root: Path) -> Dict[str, Any]:
    """The exact manifest-declared requalification records: path beneath ``root`` equal to
    the declared path, exact SHA-256, one requalified record per qualified target, complete
    fields. Never trusts a records file merely because it is well formed."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    declared = manifest.get("requalification_records") or {}
    qualified = manifest.get("qualified_targets")
    if not declared.get("path") or not declared.get("sha256") or not qualified:
        raise CohortRefused("REFUSED: manifest declares no requalification records/qualified set")
    root = Path(root).resolve()
    given = Path(prep_path).resolve()
    try:
        given.relative_to(root)
    except ValueError:
        raise CohortRefused("REFUSED: --prep is not beneath the repository root") from None
    if given != (root / declared["path"]).resolve():
        raise CohortRefused(f"REFUSED: --prep is not the manifest-declared {declared['path']}")
    data = given.read_bytes()
    if sha256_bytes(data) != declared["sha256"]:
        raise CohortRefused("REFUSED: preparation records differ from the manifest-declared hash")
    rows: Dict[str, Dict[str, Any]] = {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise CohortRefused(f"REFUSED: preparation line {number} malformed") from None
        key = row.get("key")
        if key in rows:
            raise CohortRefused(f"REFUSED: duplicate preparation record {key}")
        rows[key] = row
    if set(rows) != set(qualified):
        raise CohortRefused(f"REFUSED: preparation records {len(rows)} do not match the "
                            f"{len(qualified)} qualified targets")
    for key, row in rows.items():
        missing = [f for f in PREP_REQUIRED if row.get(f) in (None, "", {}, [])]
        if missing:
            raise CohortRefused(f"REFUSED: preparation record {key} lacks {missing}")
        if row["category"] != "requalified":
            raise CohortRefused(f"REFUSED: preparation record {key} is {row['category']!r}")
        for field in ("views", "view_manifest_sha256", "module_sha256"):
            if not all(row[field].get(label) for label in ("buggy", "fixed")):
                raise CohortRefused(f"REFUSED: preparation record {key} lacks {field}")
    return {"path": declared["path"], "sha256": declared["sha256"], "rows": rows}


# --- telemetry --------------------------------------------------------------------------------

def _number(value: Any, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(value) and value >= 0


def candidate_problems(c: Mapping[str, Any], max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    problems = []
    for field in ("raw", "raw_sha256", "module", "module_sha256", "generated_tokens",
                  "eos_reached", "finish_reason", "hit_completion_limit", "fence_stripped"):
        if field not in c:
            problems.append(f"missing {field}")
    if problems:
        return problems
    if not isinstance(c["raw"], str) or sha256_bytes(c["raw"].encode("utf-8")) != c["raw_sha256"]:
        problems.append("raw hash")
    extracted = extract(c["raw"]) if isinstance(c["raw"], str) else {}
    if extracted.get("module") != c["module"] or \
            sha256_bytes(str(c["module"]).encode("utf-8")) != c["module_sha256"]:
        problems.append("module extraction/hash")
    if extracted.get("fence_stripped") != c["fence_stripped"]:
        problems.append("fence flag")
    tokens, eos = c["generated_tokens"], c["eos_reached"]
    if not isinstance(tokens, int) or isinstance(tokens, bool) or not 0 < tokens <= max_new_tokens:
        problems.append("generated_tokens out of range")
    if not isinstance(eos, bool) or not isinstance(c["hit_completion_limit"], bool):
        problems.append("flags not boolean")
    elif eos:
        if c["finish_reason"] != "eos" or c["hit_completion_limit"]:
            problems.append("eos candidate with length finish or limit flag")
    elif c["finish_reason"] != "length" or not c["hit_completion_limit"] or \
            tokens != max_new_tokens:
        problems.append("non-EOS candidate must be a length finish at max_new_tokens")
    return problems


def row_problems(row: Mapping[str, Any], *, identity_sha256: str, arm: str, condition: str,
                 prompt_sha256: Optional[str], require_gpu_evidence: bool,
                 candidates: int = CANDIDATES, batch_size: int = BATCH_SIZE,
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 prompt_token_limit: int = PROMPT_TOKEN_LIMIT) -> List[str]:
    problems = []
    required = ("key", "arm", "condition", "seed", "target_seed", "target_key",
                "identity_sha256", "prompt_sha256", "telemetry_schema", "prompt_tokens",
                "wall_seconds", "batch_wall_seconds", "candidates_requested",
                "candidates_produced", "model_load_seconds", "peak_allocated_bytes",
                "peak_reserved_bytes", "process_peak_allocated_bytes",
                "process_peak_reserved_bytes", "candidates")
    missing = [f for f in required if f not in row]
    if missing:
        return [f"missing {missing}"]
    if row["telemetry_schema"] != TELEMETRY_SCHEMA:
        problems.append("telemetry schema")
    if row["identity_sha256"] != identity_sha256:
        problems.append("identity (stale row)")
    if row["arm"] != arm or row["condition"] != condition:
        problems.append("arm/condition")
    seed = row["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in SEEDS:
        problems.append("seed not in 42/43/44")
    if not isinstance(row["target_key"], str) or row["key"] != f"{row['target_key']}::{seed}":
        problems.append("key")
    elif not isinstance(row["target_seed"], int) or isinstance(row["target_seed"], bool) or \
            not isinstance(seed, int) or row["target_seed"] != target_seed(seed, row["target_key"]):
        problems.append("target_seed does not recompute")
    if prompt_sha256 is not None and row["prompt_sha256"] != prompt_sha256:
        problems.append("prompt hash differs from the job")
    tokens = row["prompt_tokens"]
    if not isinstance(tokens, int) or isinstance(tokens, bool) or \
            not 0 < tokens <= prompt_token_limit:
        problems.append("prompt_tokens outside 1..prompt limit")
    batches = math.ceil(candidates / batch_size)
    walls = row["batch_wall_seconds"]
    if not isinstance(walls, list) or len(walls) != batches or not all(_number(w) for w in walls):
        problems.append("batch_wall_seconds")
    elif not _number(row["wall_seconds"]) or abs(sum(walls) - row["wall_seconds"]) > 1e-3:
        problems.append("wall_seconds is not the sum of its batches")
    if row["candidates_requested"] != candidates or row["candidates_produced"] != candidates or \
            not isinstance(row["candidates"], list) or len(row["candidates"]) != candidates:
        problems.append("candidate count")
    for field in ("model_load_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
                  "process_peak_allocated_bytes", "process_peak_reserved_bytes"):
        if not _number(row[field], allow_none=not require_gpu_evidence):
            problems.append(f"{field} missing or invalid")
    if require_gpu_evidence and all(_number(row[f]) for f in (
            "peak_allocated_bytes", "process_peak_allocated_bytes")) and \
            row["peak_allocated_bytes"] > row["process_peak_allocated_bytes"]:
        problems.append("row peak exceeds the process peak")
    for i, c in enumerate(row["candidates"] if isinstance(row["candidates"], list) else []):
        problems += [f"candidate {i}: {p}" for p in candidate_problems(c, max_new_tokens)]
    return problems


# --- per-arm loading ----------------------------------------------------------------------------

def arm_paths(root: Optional[Path], base: Optional[Path], sft: Optional[Path],
              condition: str) -> Dict[str, Dict[str, Path]]:
    """Either a generation root containing base/ and sft/, or explicit per-arm directories."""
    explicit = base is not None or sft is not None
    if (root is not None and explicit) or (root is None and (base is None or sft is None)):
        raise CohortRefused("REFUSED: give --generations ROOT or both --base-generations and "
                            "--sft-generations")
    dirs = {"base": Path(root) / "base", "sft": Path(root) / "sft"} if root is not None else \
        {"base": Path(base), "sft": Path(sft)}
    if dirs["base"].resolve() == dirs["sft"].resolve():
        raise CohortRefused("REFUSED: base and sft generation directories are the same")
    return {arm: {"dir": d, "rows": d / f"generations_{condition}_{arm}.jsonl",
                  "contract": d / f"contract_{condition}_{arm}.json"} for arm, d in dirs.items()}


def _arm_contract(p: Mapping[str, Path], arm: str, cohort: Mapping[str, Any]) -> Dict[str, Any]:
    if not p["rows"].is_file() or not p["contract"].is_file():
        raise CohortRefused(f"REFUSED: {arm} generation file or contract missing")
    contract = json.loads(p["contract"].read_text(encoding="utf-8"))
    identity = contract.get("identity") or {}
    problems = []
    if identity.get("arm") != arm:
        problems.append(f"contract arm is {identity.get('arm')!r} (swapped arms?)")
    if identity.get("condition") != cohort["condition"]:
        problems.append("condition")
    if identity.get("job_sha256") != cohort["job_sha256"] or \
            identity.get("job_file_sha256") != cohort["job_file_sha256"]:
        problems.append("job differs")
    if contract.get("telemetry_schema") != TELEMETRY_SCHEMA:
        problems.append("telemetry schema")
    adapter = identity.get("adapter_manifest_sha256")
    if (arm == "base") != (adapter is None):
        problems.append("adapter identity inconsistent with the arm")
    if problems:
        raise CohortRefused(f"REFUSED: {arm} generation contract: {problems}")
    return contract


def _arm_rows(p: Mapping[str, Path], arm: str, contract: Mapping[str, Any],
              cohort: Mapping[str, Any], gpu: bool) -> Dict[str, Dict[str, Any]]:
    expected = {f"{t}::{s}" for t in cohort["generation"] for s in SEEDS}
    data = p["rows"].read_bytes()
    if data and not data.endswith(b"\n"):
        raise CohortRefused(f"REFUSED: {arm} generation file ends with a partial line")
    ihash = contract_sha(contract)
    seen: Dict[str, Dict[str, Any]] = {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            raise CohortRefused(f"REFUSED: {arm} line {number} malformed") from None
        key = row.get("key")
        if key not in expected:
            raise CohortRefused(f"REFUSED: {arm} line {number}: unexpected target-seed {key!r}")
        if key in seen:
            raise CohortRefused(f"REFUSED: {arm} line {number}: duplicate {key}")
        problems = row_problems(row, identity_sha256=ihash, arm=arm,
                                condition=cohort["condition"],
                                prompt_sha256=cohort["items"][row["target_key"]]["prompt_sha256"],
                                require_gpu_evidence=gpu)
        if problems:
            raise CohortRefused(f"REFUSED: {arm} {key}: {problems[:4]}")
        seen[key] = row
    if set(seen) != expected:
        raise CohortRefused(f"REFUSED: {arm} generations incomplete: "
                            f"{len(expected - set(seen))} target-seed rows missing")
    return seen


def verify_single_arm(arm_dir: Path, arm: str, cohort: Mapping[str, Any]) -> Dict[str, Any]:
    """One arm complete and exact (amendment v2.4 I.2): every expected target-seed row,
    full telemetry, the frozen job; returns immutable hashes. Never copies or merges."""
    d = Path(arm_dir)
    p = {"dir": d, "rows": d / f"generations_{cohort['condition']}_{arm}.jsonl",
         "contract": d / f"contract_{cohort['condition']}_{arm}.json"}
    contract = _arm_contract(p, arm, cohort)
    rows = _arm_rows(p, arm, contract, cohort, contract["identity"].get("backend") == "hf")
    return {"arm": arm, "rows": len(rows),
            "candidates": sum(len(r["candidates"]) for r in rows.values()),
            "file_sha256": sha256_file(p["rows"]), "contract_sha256": sha256_file(p["contract"]),
            "identity_sha256": contract_sha(contract), "identity": contract["identity"],
            "telemetry_schema": contract["telemetry_schema"]}


def load_arm_generations(paths: Mapping[str, Mapping[str, Path]], cohort: Mapping[str, Any]
                         ) -> Dict[str, Any]:
    contracts = {arm: _arm_contract(paths[arm], arm, cohort) for arm in ARMS}
    out = {"rows": {}, "files_sha256": {}, "contracts_sha256": {}}
    comparable = [{**{k: v for k, v in c.items() if k != "identity"},
                   "identity": {k: v for k, v in c["identity"].items() if k not in ARM_SPECIFIC}}
                  for c in (contracts["base"], contracts["sft"])]
    if comparable[0] != comparable[1]:
        diff = sorted(k for k in set(comparable[0]["identity"]) | set(comparable[1]["identity"])
                      if comparable[0]["identity"].get(k) != comparable[1]["identity"].get(k))
        raise CohortRefused(f"REFUSED: base and sft generation contracts differ beyond the arm: "
                            f"{diff or 'generator/contract'}")
    gpu = contracts["base"]["identity"].get("backend") == "hf"
    for arm in ARMS:
        seen = _arm_rows(paths[arm], arm, contracts[arm], cohort, gpu)
        for row in seen.values():
            out["rows"][(arm, row["target_key"], int(row["seed"]))] = row
        out["files_sha256"][arm] = sha256_file(paths[arm]["rows"])
        out["contracts_sha256"][arm] = sha256_file(paths[arm]["contract"])
    out["identity_sha256"] = {arm: contract_sha(contracts[arm]) for arm in ARMS}
    out["gpu_evidence_required"] = gpu
    return out
