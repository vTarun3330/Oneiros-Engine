"""Acquisition pipeline tests: mocked GitHub REST and a real local git fixture."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from harness.github_acquisition import (
    AcquisitionFailure, ApiClient, ContentStore, HttpResponse, IntegrityViolation, Journal,
    LocalGitRepository, RateLimiter, RecordingObjects, resolve_repository,
)
from harness.repository_isolation import ApiResponse, git_object_id, parse_commit, parse_tree
from harness.repository_native_candidates import (
    RepositoryContext, build_candidate, classify_commit, evaluate_candidate, is_test_path,
    licence_text_problems, store_candidate,
)
from tests.test_repository_isolation import _freeze, _write_fake_root

API = "https://api.github.com/repos/"
MIT = ("MIT License\n\nCopyright (c) 2025 Example\n\nPermission is hereby granted, free of "
       "charge, to any person obtaining a copy of this software.\n")
MERGE_BUGGY = ("def merge_ranges(pairs):\n    pairs = sorted(pairs)\n    merged = [pairs[0]]\n"
               "    for lo, hi in pairs[1:]:\n        if lo < merged[-1][1]:\n"
               "            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))\n"
               "        else:\n            merged.append((lo, hi))\n    return merged\n")
CLAMP = ("def clamp_all(values, low, high):\n    out = []\n    for value in values:\n"
         "        if value < low:\n            out.append(low)\n        elif value > high:\n"
         "            out.append(high)\n        else:\n            out.append(value)\n"
         "    return out\n")
WINDOW = ("def window_totals(values, size):\n    totals = []\n"
          "    for start in range(0, len(values) - size):\n"
          "        totals.append(sum(values[start:start + size]))\n    return totals\n")
UTIL = ("def scale_all(values, factor):\n    result = []\n    for value in values:\n"
        "        result.append(value * factor + 0)\n    return result\n")


# --- the local git fixture ---------------------------------------------------------------

def _git(repo: Path, *args: str, date: str | None = None) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="Dev", GIT_AUTHOR_EMAIL="dev@example.invalid",
               GIT_COMMITTER_NAME="Dev", GIT_COMMITTER_EMAIL="dev@example.invalid")
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    result = subprocess.run(["git", "-C", str(repo), *args], env=env, capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str, date: str) -> str:
    for path, text in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, date=date)
    return _git(repo, "rev-parse", "HEAD")


def _core(merge: str = MERGE_BUGGY, clamp: str = CLAMP, window: str = WINDOW) -> str:
    return f"{merge}\n\n{clamp}\n\n{window}"


@pytest.fixture(scope="module")
def source_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("source") / "ranges"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for key, value in (("core.autocrlf", "false"), ("uploadpack.allowFilter", "true"),
                       ("uploadpack.allowAnySHA1InWant", "true")):
        _git(repo, "config", key, value)
    commits = {}
    commits["initial"] = _commit(repo, {"LICENSE": MIT, "src/ranges/core.py": _core(),
                                        "src/ranges/util.py": UTIL,
                                        "tests/test_core.py": "def test_x():\n    assert 1\n"},
                                 "initial", "2024-11-20T10:00:00+00:00")
    commits["pre_cutoff"] = _commit(repo, {"src/ranges/core.py": _core(
        window=WINDOW.replace("len(values) - size)", "len(values) - size + 1)"))},
        "Fix window bound (#5)", "2024-12-15T10:00:00+00:00")
    core_now = _core(window=WINDOW.replace("len(values) - size)", "len(values) - size + 1)"))
    fixed_merge = MERGE_BUGGY.replace("lo < merged", "lo <= merged")
    commits["single"] = _commit(repo, {"src/ranges/core.py": core_now.replace(
        MERGE_BUGGY, fixed_merge)}, "Fix overlap check in merge_ranges (#7)",
        "2025-02-01T10:00:00+00:00")
    core_now = core_now.replace(MERGE_BUGGY, fixed_merge)
    commits["multi"] = _commit(repo, {
        "src/ranges/core.py": core_now.replace("value > high", "value >= high"),
        "src/ranges/util.py": UTIL.replace("+ 0", "")}, "Fix clamp and scale together (#8)",
        "2025-02-10T10:00:00+00:00")
    core_now = core_now.replace("value > high", "value >= high")
    commits["with_test"] = _commit(repo, {
        "src/ranges/core.py": core_now.replace("value < low", "value <= low"),
        "tests/test_core.py": "def test_x():\n    assert 2\n"},
        "Fix lower clamp and add a regression test (#9)", "2025-02-20T10:00:00+00:00")
    core_now = core_now.replace("value < low", "value <= low")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, {"src/ranges/core.py": core_now.replace("totals = []", "totals = [0]")},
            "fix totals start", "2025-02-25T10:00:00+00:00")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"README.md": "docs\n"}, "docs: update readme", "2025-02-26T10:00:00+00:00")
    _git(repo, "merge", "-q", "--no-ff", "feature", "-m",
         "Merge pull request #10 from dev/feature\n\nfix totals start",
         date="2025-03-01T10:00:00+00:00")
    commits["merge"] = _git(repo, "rev-parse", "HEAD")
    commits["refactor"] = _commit(repo, {"src/ranges/util.py": UTIL.replace("result", "scaled")},
                                  "Rename a local variable", "2025-03-05T10:00:00+00:00")
    return repo, commits


# --- a mocked GitHub REST API --------------------------------------------------------------

def _metadata(full_name="example-org/ranges", fork_parent=None, spdx="MIT", repo_id=4242):
    value = {"full_name": full_name, "id": repo_id, "node_id": "R_examplenode01",
             "html_url": f"https://github.com/{full_name}", "fork": fork_parent is not None,
             "license": {"spdx_id": spdx}, "default_branch": "main", "archived": False}
    if fork_parent:
        value["parent"] = {"full_name": fork_parent, "id": 7, "node_id": "R_parentnode01"}
    return value


def _pull(number, full_name="example-org/ranges"):
    return {"url": f"{API}{full_name}/pulls/{number}", "number": number,
            "base": {"repo": {"full_name": full_name}}, "merged_at": "2025-02-01T10:00:00Z"}


class FakeTransport:
    def __init__(self, routes: dict[str, list[HttpResponse] | HttpResponse]):
        self.routes = routes
        self.calls: list[str] = []
        self.explode_on: str | None = None

    def get(self, url, headers):
        self.calls.append(url)
        if url == self.explode_on:
            raise SystemExit("simulated process death")
        route = self.routes.get(url)
        if route is None:
            return HttpResponse(404, {}, b'{"message": "Not Found"}', url)
        if isinstance(route, list):
            return route.pop(0) if len(route) > 1 else route[0]
        return route


def ok(value, etag='"e1"'):
    return HttpResponse(200, {"etag": etag}, json.dumps(value).encode("utf-8"), "")


class FakeClock:
    def __init__(self):
        self.now = 1_760_000_000.0
        self.slept: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _client(routes, clock=None):
    clock = clock or FakeClock()
    transport = FakeTransport(routes)
    return ApiClient(transport, RateLimiter(min_interval=0, clock=clock.clock, sleep=clock.sleep),
                     backoff=lambda attempt: 1.0), transport, clock


def _routes(metadata=None):
    routes = {f"{API}example-org/ranges": ok(metadata or _metadata())}
    for number in (7, 8, 9, 10):
        routes[f"{API}example-org/ranges/pulls/{number}"] = ok(_pull(number))
    return routes


# --- unit: store, journal, HTTP ----------------------------------------------------------

def test_content_store_never_rewrites_and_detects_corruption(tmp_path):
    store = ContentStore(tmp_path)
    digest = store.put_raw(b"payload")
    assert store.put_raw(b"payload") == digest
    path = tmp_path / "raw" / digest[:2] / digest
    path.write_bytes(b"tampered")
    with pytest.raises(IntegrityViolation):
        store.get_raw(digest)
    with pytest.raises(IntegrityViolation):
        store.put_raw(b"payload")            # would rewrite a different existing object
    oid = store.put_git("blob", b"x = 1\n")
    assert store.get_git(oid) == ("blob", b"x = 1\n")
    with pytest.raises(AcquisitionFailure) as failure:
        store.put_git("blob", b"x = 2\n", expected_oid=oid)
    assert failure.value.category == "hash_mismatch"


def test_journal_is_durable_append_only_and_tolerates_a_torn_line(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.record("cand:a", {"status": "done"})
    with open(tmp_path / "journal.jsonl", "a", encoding="utf-8") as stream:
        stream.write('{"key": "cand:b", "trunc')             # a crash mid-write
    reopened = Journal(tmp_path / "journal.jsonl")
    assert reopened.done("cand:a") and not reopened.done("cand:b")
    with pytest.raises(IntegrityViolation):
        reopened.record("cand:a", {"status": "again"})


def test_rate_limit_headers_are_honoured():
    reset_in = 120
    clock = FakeClock()
    exhausted = HttpResponse(403, {"x-ratelimit-remaining": "0",
                                   "x-ratelimit-reset": str(int(clock.now) + reset_in)}, b"{}", "")
    client, transport, _ = _client({f"{API}a/b": [exhausted, ok({"x": 1})]}, clock)
    response, _ = client.get(f"{API}a/b")
    assert json.loads(response.body) == {"x": 1} and response.etag == '"e1"'
    assert sum(clock.slept) >= reset_in and client.retries == 1
    client, _, clock = _client({f"{API}a/b": [HttpResponse(429, {"retry-after": "7"}, b"", ""),
                                              ok({"x": 2})]})
    client.get(f"{API}a/b")
    assert 7 in clock.slept


def test_transient_failures_retry_with_backoff_then_classify():
    client, _, clock = _client({f"{API}a/b": [HttpResponse(502, {}, b"", ""), ok({"x": 1})]})
    assert json.loads(client.get(f"{API}a/b")[0].body) == {"x": 1} and clock.slept == [1.0]
    client, _, _ = _client({f"{API}a/b": HttpResponse(503, {}, b"", "")})
    with pytest.raises(AcquisitionFailure) as failure:
        client.get(f"{API}a/b")
    assert failure.value.category == "network_error"


@pytest.mark.parametrize("body, category", [(b"{not json", "corrupt_response"),
                                            (b"[1, 2]", "corrupt_response")])
def test_corrupt_responses_are_classified(body, category):
    client, _, _ = _client({f"{API}a/b": HttpResponse(200, {}, body, "")})
    with pytest.raises(AcquisitionFailure) as failure:
        client.get(f"{API}a/b")
    assert failure.value.category == category


def test_not_found_and_redirects_are_never_followed_silently():
    client, _, _ = _client({})
    with pytest.raises(AcquisitionFailure) as failure:
        client.get(f"{API}missing/repo")
    assert failure.value.category == "not_found"
    moved = HttpResponse(301, {"location": "https://api.github.com/repositories/42"}, b"", "")
    client, _, _ = _client({f"{API}old/name": moved})
    with pytest.raises(AcquisitionFailure) as failure:
        client.get(f"{API}old/name")
    assert failure.value.category == "redirected"


def test_a_renamed_repository_is_resolved_by_requerying_and_recorded():
    moved = HttpResponse(301, {"location": "https://api.github.com/repositories/42"}, b"", "")
    client, transport, _ = _client({
        f"{API}old-org/ranges": moved,
        "https://api.github.com/repositories/42": ok(_metadata("example-org/ranges")),
        f"{API}example-org/ranges": ok(_metadata("example-org/ranges"))})
    response, value, renames = resolve_repository(client, "Old-Org/Ranges")
    assert value["full_name"] == "example-org/ranges"
    assert response.url == f"{API}example-org/ranges"        # queried == returned
    assert renames == [{"queried": "old-org/ranges", "redirected_to": "example-org/ranges"}]


def test_commit_classification_and_test_paths():
    assert classify_commit("Fix overlap (#7)") == (True, "fix_candidate", 7)
    assert classify_commit("docs: fix typo (#3)")[1] == "non_code_commit_subject"
    assert classify_commit("Rename a variable (#4)")[1] == "not_described_as_a_fix"
    assert classify_commit("Fix thing")[1] == "no_linked_pr_or_issue"
    assert classify_commit("Merge pull request #10 from x/y\n\nfix bound")[1] == \
        "fix_merge_commit"
    assert is_test_path("tests/test_core.py") and is_test_path("pkg/conftest.py")
    assert not is_test_path("src/ranges/core.py")
    assert licence_text_problems("MIT", MIT) == []
    assert "licence_text_does_not_match_spdx" in licence_text_problems("MIT", "All rights reserved")
    assert "licence_not_admitted_by_d2" in licence_text_problems("GPL-3.0", MIT)


# --- the pipeline on a real local git repository -------------------------------------------

@pytest.fixture(scope="module")
def fetched(source_repo, tmp_path_factory):
    repo, commits = source_repo
    git = LocalGitRepository(tmp_path_factory.mktemp("clone") / "ranges.git", repo.as_uri())
    git.fetch_since("2024-11-01", "main")
    return git, commits


@pytest.fixture(scope="module")
def frozen_universe(tmp_path_factory):
    root = tmp_path_factory.mktemp("universe") / "repo"
    _write_fake_root(root)
    return _freeze(root)[0]


def _context(metadata=None):
    value = metadata or _metadata()
    body = json.dumps(value).encode("utf-8")
    import hashlib
    return RepositoryContext("example-org/ranges", ApiResponse(
        url=f"{API}example-org/ranges", body=body, sha256=hashlib.sha256(body).hexdigest(),
        retrieved_utc="2026-09-25T00:00:00Z", etag='"e"'), value, [])


def _evaluate(fetched, frozen_universe, tmp_path, key, metadata=None, source=None):
    git, commits = fetched
    client, _, _ = _client(_routes(metadata) | {f"{API}example-org/ranges/pulls/5": ok(_pull(5))})
    objects = RecordingObjects(source or git)
    number = {"single": 7, "multi": 8, "with_test": 9, "merge": 10, "pre_cutoff": 5}[key]
    candidate, info = build_candidate(_context(metadata), commits[key], number, objects, client)
    if candidate is None:
        return None, info
    store = ContentStore(tmp_path / "store")
    addresses = store_candidate(store, candidate)
    return evaluate_candidate(candidate, frozen_universe, store, addresses,
                              info["licence_text"]), info


def test_an_exact_single_file_fix_is_admitted(fetched, frozen_universe, tmp_path):
    outcome, info = _evaluate(fetched, frozen_universe, tmp_path, "single")
    assert outcome["admitted"] is True, outcome["reasons"]
    assert info["changed_files"] == ["src/ranges/core.py"]
    assert info["target_qualname"] == "merge_ranges"
    assert outcome["revalidation_valid"] is True
    assert outcome["derived"]["submitted_patch_matches_derived_diff"] is True


def test_policy_a_prime_on_real_git_objects(fetched, frozen_universe, tmp_path):
    # Two production files changed: refused.
    outcome, info = _evaluate(fetched, frozen_universe, tmp_path, "multi")
    assert len(info["changed_files"]) == 2 and outcome["admitted"] is False
    assert ("insufficient_isolation_evidence:authentication:changed_production_source_outside_"
            "target:src/ranges/util.py") in outcome["reasons"]
    # The target file plus its regression test: admitted, test evidence recorded.
    outcome, info = _evaluate(fetched, frozen_universe, tmp_path, "with_test")
    assert outcome["admitted"] is True, outcome["reasons"]
    assert info["changed_file_categories"] == {"src/ranges/core.py": "production_source",
                                               "tests/test_core.py": "test"}
    auxiliary = outcome["derived"]["auxiliary_changes"]
    assert [item["path"] for item in auxiliary] == ["tests/test_core.py"]
    assert outcome["derived"]["regression_test_changed"] is True


def test_a_merge_commit_is_refused(fetched, frozen_universe, tmp_path):
    outcome, _ = _evaluate(fetched, frozen_universe, tmp_path, "merge")
    assert any("fixed_commit_is_not_a_direct_single_parent_commit" in reason
               for reason in outcome["reasons"])


def test_a_fix_before_the_temporal_cutoff_is_refused(fetched, frozen_universe, tmp_path):
    outcome, _ = _evaluate(fetched, frozen_universe, tmp_path, "pre_cutoff")
    assert any("fixed_commit_before_temporal_cutoff" in reason for reason in outcome["reasons"])


def test_fork_identity_is_checked_against_the_universe(fetched, frozen_universe, tmp_path):
    outcome, _ = _evaluate(fetched, frozen_universe, tmp_path, "single",
                           metadata=_metadata(fork_parent="demo-org/demo"))
    assert "fork_parent_in_reference_universe:demo-org/demo" in outcome["reasons"]


class _HidingSource:
    def __init__(self, inner, hidden=(), tampered=()):
        self.inner, self.hidden, self.tampered = inner, set(hidden), set(tampered)

    def read(self, oid):
        if oid in self.hidden:
            return None
        found = self.inner.read(oid)
        if found and oid in self.tampered:
            return found[0], found[1] + b"\n# tampered\n"
        return found


def _subtree(git, commit, name):
    tree = parse_commit(git.read(commit)[1])["tree"]
    return parse_tree(git.read(tree)[1])[name][1]


def test_missing_tree_evidence_is_an_acquisition_failure(fetched, frozen_universe, tmp_path):
    git, commits = fetched
    hidden = _subtree(git, commits["single"], "src")
    with pytest.raises(AcquisitionFailure) as failure:
        _evaluate(fetched, frozen_universe, tmp_path, "single",
                  source=_HidingSource(git, hidden=[hidden]))
    assert failure.value.category == "incomplete_tree_evidence"


def test_object_bytes_that_do_not_hash_to_their_id_are_refused(fetched, frozen_universe,
                                                               tmp_path):
    git, commits = fetched
    tree = parse_commit(git.read(commits["single"])[1])["tree"]
    with pytest.raises(AcquisitionFailure) as failure:
        _evaluate(fetched, frozen_universe, tmp_path, "single",
                  source=_HidingSource(git, tampered=[tree]))
    assert failure.value.category == "hash_mismatch"


def test_stored_evidence_is_re_hashed_on_read(fetched, frozen_universe, tmp_path):
    outcome, _ = _evaluate(fetched, frozen_universe, tmp_path, "single")
    store = ContentStore(tmp_path / "store")
    sidecar = outcome and next((tmp_path / "store" / "raw").rglob("*"))
    assert sidecar is not None
    for obj in (tmp_path / "store" / "git").rglob("*"):
        if obj.is_file():
            store.get_git(obj.name)


# --- the durable pilot runner: interruption and resume -------------------------------------

def _pilot_config(tmp_path, repo_list):
    listing = tmp_path / "repositories.json"
    listing.write_text(json.dumps({"repositories": repo_list}), encoding="utf-8")
    return {"label": "fixture pilot", "repositories_file": str(listing),
            "store": str(tmp_path / "pilot"), "report": str(tmp_path / "report.json"),
            "per_repository_cap": 10, "max_candidates": 100, "min_repositories": 1,
            "fetch_since": "2024-11-01", "candidates_since": "2025-01-01"}


def test_pilot_resumes_after_a_process_death_without_rewriting(source_repo, frozen_universe,
                                                               tmp_path):
    from scripts import run_repository_native_acquisition_pilot as pilot
    repo, commits = source_repo
    store = tmp_path / "pilot"
    config = _pilot_config(tmp_path, ["example-org/ranges"])
    first, transport, _ = _client(_routes())
    # Candidates are processed newest first: #10 (merge), #9, #8, #7.  Die at #8.
    transport.explode_on = f"{API}example-org/ranges/pulls/8"
    kwargs = dict(frozen=frozen_universe, git_url=lambda repository: repo.as_uri())
    with pytest.raises(SystemExit):
        pilot.run(store, config, client=first, **kwargs)
    journal = Journal(store / "journal.jsonl")
    done_before = {entry["key"] for entry in journal.entries()}
    assert f"cand:example-org/ranges@{commits['with_test']}" in done_before
    assert f"cand:example-org/ranges@{commits['multi']}" not in done_before
    second, transport2, _ = _client(_routes())
    assert pilot.run(store, config, client=second, **kwargs) == 0
    # Resumed work only: no repository or already-journaled candidate is fetched again.
    assert f"{API}example-org/ranges" not in transport2.calls
    assert f"{API}example-org/ranges/pulls/9" not in transport2.calls
    assert f"{API}example-org/ranges/pulls/10" not in transport2.calls
    assert f"{API}example-org/ranges/pulls/8" in transport2.calls
    entries = Journal(store / "journal.jsonl").entries()
    keys = [entry["key"] for entry in entries]
    assert len(keys) == len(set(keys))
    identity = {"bundle_generation": "fixture", "bundle_manifest_sha256": "b" * 64,
                "reference_universe_receipt_sha256": frozen_universe.receipt_sha256}
    report = pilot.build_report(store, config, None, frozen_identity=identity, label="fixture")
    counts = report["counts"]
    assert counts["candidates_inspected"] == 4        # #7, #8, #9 and the merge (#10)
    assert counts["admitted"] == 2                    # #7 alone and #9 with its test
    assert counts["other_production_source_exclusions"] == 1
    assert counts["admitted_with_authenticated_regression_test"] == 1
    assert report["store_verification"]["problems"] == []
    assert report["events"]["protected_data_access"] is False
    assert report["gate"] == {"no_protected_data_access": True}
    assert report["identity"]["api"]["sessions"] == 1          # the dead session left no entry
    assert sorted(item["target"] for item in report["admitted"]) == ["clamp_all", "merge_ranges"]
    from harness.acquisition_receipt import validate_receipt
    assert validate_receipt(report) == []
    pilot.publish(report, tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text())["identity"]["journal_sha256"]
