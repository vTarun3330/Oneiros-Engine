from pathlib import Path

from utils.reproducibility import build_reproducibility_manifest, source_tree_sha256


def test_source_hash_is_stable_and_manifest_records_model():
    project_root = Path(__file__).resolve().parent.parent
    assert source_tree_sha256(project_root) == source_tree_sha256(project_root)
    manifest = build_reproducibility_manifest(project_root, "model", "commit123")
    assert len(manifest["source_tree_sha256"]) == 64
    assert len(manifest["dependency_spec_sha256"]) == 64
    assert manifest["model_name"] == "model"
    assert manifest["model_revision"] == "commit123"
    assert "transformers" in manifest["runtime_dependencies"]
