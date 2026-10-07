"""Immutable arm registry for the v2.7 three-arm repository-native evaluation (stdlib only).

Every arm of the study is declared HERE, with the exact adapter directory and every identity
that resolves it, frozen before any model output exists. Nothing accepts an arbitrary adapter
path: the generator, launch gate, generation loader, executor contract and analysis bind the
canonical registry hash, and ``verify`` recomputes every identity from disk.

- base:    the pinned base model, no adapter;
- sft:     A@431 (results/v4_2_development_selection_receipt.json, arms["A@431"]);
- relearn: relearning checkpoint-141 (results/v4_2_relearning_receipt.json ``adapter``; adapter
           SHA-256 bound by results/v4_2_failure_taxonomy_relearning.json).

Receipt hashes are canonical (CRLF normalised to LF) so a checkout reproduces them; adapter files
are local, git-ignored binaries and are hashed byte-exact (machine-bound by design).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

REGISTRY_SCHEMA = "oneiros_native_arm_registry_v1"
REGISTRY_NAME = "oneiros_v27_three_arm_registry_r5"
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASE_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
ARMS = ("base", "sft", "relearn")

REGISTRY: Dict[str, Any] = {
    "schema_version": REGISTRY_SCHEMA, "name": REGISTRY_NAME,
    "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
    "arms": {
        "base": {"label": "base", "adapter_dir": None},
        "sft": {
            "label": "A@431",
            "adapter_dir": "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter",
            "adapter_manifest_sha256":
                "82f24fed40316249f307fc6b835f2d1fa9d1b7ada61e72e9892fb397e377f2a0",
            "adapter_model_sha256":
                "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7",
            "adapter_config_sha256":
                "4dda431a828e688fa59e8c986981773c5fdcf67de50da5e338aa6a314f00f5a5",
            "receipts": {
                "selection": {"path": "results/v4_2_development_selection_receipt.json",
                              "canonical_sha256": "0188324fe9eb2a51eebef71bbd4cc3ed45584f12393bf0fa6539817a09e58969",
                              "pointer": "/arms/A@431"}}},
        "relearn": {
            "label": "relearning checkpoint-141",
            "adapter_dir": "checkpoints/local_sft_relearn_v2_seed42/sft_validation_best/checkpoint-141",
            "adapter_manifest_sha256":
                "c1dcea74c9f97b012670114fe33e7695ad478eb30600e77f214f37781be4e133",
            "adapter_model_sha256":
                "af262f4f8b43bab5426df6bf2556769818ca6f8e5cf8c85412238962ea659e63",
            "adapter_config_sha256":
                "6d47a3e56de5fd27dcbd0d90f7347643a14f40e4a1145dd47b590e2140577e26",
            "receipts": {
                "selection": {"path": "results/v4_2_relearning_receipt.json",
                              "canonical_sha256": "e9a5379de51d9e7f6bb8efdd8077925c8b2631c50b87905dcadca1a737b52098",
                              "pointer": "/adapter"},
                "hash_binding": {"path": "results/v4_2_failure_taxonomy_relearning.json",
                                 "canonical_sha256": "9c4f635a4e9ec3504971c247d31f6624e9cd892608a14b3802c175a263d9d902",
                                 "pointer": "/artifacts/*/adapter_sha256"}}},
    },
}


class RegistryRefused(SystemExit):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(path: Path) -> str:
    return _sha(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def registry_sha256(registry: Mapping[str, Any] = REGISTRY) -> str:
    return _sha(json.dumps(registry, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def adapter_manifest(directory: Path) -> Dict[str, str]:
    """Every file in the adapter directory with its SHA-256 (same rule as the generator)."""
    return {p.relative_to(directory).as_posix(): _sha(p.read_bytes())
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def manifest_sha256(manifest: Mapping[str, str]) -> str:
    return _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def entry(arm: str, registry: Mapping[str, Any] = REGISTRY) -> Mapping[str, Any]:
    if arm not in registry["arms"]:
        raise RegistryRefused(f"REFUSED: arm {arm!r} is not in the frozen registry")
    return registry["arms"][arm]


def adapter_dir(root: Path, arm: str, registry: Mapping[str, Any] = REGISTRY):
    rel = entry(arm, registry)["adapter_dir"]
    return None if rel is None else Path(root) / rel


def verify_arm(root: Path, arm: str, registry: Mapping[str, Any] = REGISTRY) -> List[str]:
    """Recompute every identity of one arm from disk; an empty list means verified."""
    e = entry(arm, registry)
    if e["adapter_dir"] is None:
        return [] if arm == "base" else [f"{arm}: a trained arm without an adapter"]
    if arm == "base":
        return ["base: the base arm must not carry an adapter"]
    problems = []
    d = Path(root) / e["adapter_dir"]
    if not d.is_dir():
        return [f"{arm}: adapter directory missing"]
    files = adapter_manifest(d)
    if manifest_sha256(files) != e["adapter_manifest_sha256"]:
        problems.append(f"{arm}: adapter directory manifest differs (modified or swapped)")
    if files.get("adapter_model.safetensors") != e["adapter_model_sha256"]:
        problems.append(f"{arm}: adapter weights differ")
    if files.get("adapter_config.json") != e["adapter_config_sha256"]:
        problems.append(f"{arm}: adapter config differs")
    try:
        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
        if cfg.get("base_model_name_or_path") != registry["base_model"]:
            problems.append(f"{arm}: adapter was not trained on the registered base model")
    except (OSError, ValueError):
        problems.append(f"{arm}: adapter config unreadable")
    for name, rec in e["receipts"].items():
        path = Path(root) / rec["path"]
        if not path.is_file() or canonical_sha256(path) != rec["canonical_sha256"]:
            problems.append(f"{arm}: {name} receipt missing or changed")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if arm == "sft" and name == "selection":
            a = data["arms"]["A@431"]
            if a["adapter_path"] != f"{e['adapter_dir']}/adapter_model.safetensors" or \
                    a["adapter_sha256"] != e["adapter_model_sha256"]:
                problems.append("sft: selection receipt names a different adapter")
        if arm == "relearn" and name == "selection":
            named = str(data["adapter"]).replace("\\", "/")
            if not named.endswith("/" + e["adapter_dir"]):
                problems.append("relearn: selection receipt names a different adapter")
        if arm == "relearn" and name == "hash_binding":
            bound = {a.get("adapter_sha256") for a in data.get("artifacts", [])} - {None}
            if bound != {e["adapter_model_sha256"]}:
                problems.append("relearn: hash-binding receipt binds a different adapter")
    return problems


def verify(root: Path, arms=ARMS, registry: Mapping[str, Any] = REGISTRY) -> List[str]:
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        return ["registry schema"]
    labels = [registry["arms"][a]["label"] for a in registry["arms"]]
    dirs = [registry["arms"][a]["adapter_dir"] for a in registry["arms"]
            if registry["arms"][a]["adapter_dir"]]
    problems = []
    if len(set(labels)) != len(labels) or len(set(dirs)) != len(dirs):
        problems.append("registry labels or adapter directories are not unique")
    for arm in arms:
        problems += verify_arm(root, arm, registry)
    return problems


def summary(registry: Mapping[str, Any] = REGISTRY) -> Dict[str, Any]:
    """What a manifest, preflight or authorisation binds."""
    return {"schema_version": registry["schema_version"], "name": registry["name"],
            "sha256": registry_sha256(registry), "arms": list(registry["arms"]),
            "adapter_manifest_sha256": {a: registry["arms"][a].get("adapter_manifest_sha256")
                                        for a in registry["arms"]}}
