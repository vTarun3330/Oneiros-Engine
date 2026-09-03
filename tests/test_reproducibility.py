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


def test_functional_identity_excludes_git_commit_but_keeps_source():
    """Committing a run's own results must not invalidate its adapter.

    Section 47 requires results to be committed. Comparing whole
    reproducibility manifests also compared git_commit, so doing what section
    47 demands invalidated the adapter that produced those results. Source
    identity is the stronger guarantee: it catches uncommitted edits that a
    commit SHA cannot.
    """
    from utils.reproducibility import FUNCTIONAL_IDENTITY_FIELDS, functional_identity

    trained = {
        "git_commit": "aaaaaaaa",
        "git_dirty": False,
        "source_tree_sha256": "SOURCE-1",
        "dependency_spec_sha256": "DEPS-1",
        "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "model_revision": "main",
        "python_version": "3.12.14",
        "python_implementation": "CPython",
        "runtime_dependencies": {"torch": "2.5.1+cu124"},
    }
    after_committing_results = dict(trained, git_commit="bbbbbbbb")

    assert functional_identity(trained) == functional_identity(after_committing_results)
    assert "git_commit" not in FUNCTIONAL_IDENTITY_FIELDS


def test_functional_identity_still_rejects_changed_source_or_model():
    """The gate must still refuse an adapter built by different code."""
    from utils.reproducibility import functional_identity

    trained = {
        "source_tree_sha256": "SOURCE-1",
        "dependency_spec_sha256": "DEPS-1",
        "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "model_revision": "main",
        "python_version": "3.12.14",
        "python_implementation": "CPython",
        "runtime_dependencies": {"torch": "2.5.1+cu124"},
    }

    assert functional_identity(dict(trained, source_tree_sha256="SOURCE-2")) != functional_identity(trained)
    assert functional_identity(dict(trained, model_name="microsoft/Phi-3-mini-4k-instruct")) != functional_identity(trained)
    assert functional_identity(dict(trained, runtime_dependencies={"torch": "9.9.9"})) != functional_identity(trained)
    assert functional_identity(dict(trained, dependency_spec_sha256="DEPS-2")) != functional_identity(trained)
