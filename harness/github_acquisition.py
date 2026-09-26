"""Read-only, resumable, content-addressed acquisition of candidate-bug evidence.

Approved by decision D1.  For each approved repository it acquires, and binds
by hash, exactly the evidence ``harness.repository_isolation`` authenticates:

* repository and issue/PR API responses: raw bytes, SHA-256, UTC retrieval
  time and ETag (GitHub REST, read-only GETs);
* the buggy and fixed commit objects, every tree object the changed-file proof
  and the licence/target path resolution read, and the target and licence
  blobs - fetched with git over HTTPS into a local blob-less bare repository
  and read back as raw loose-object bodies, each re-hashed to its object ID.

Nothing submitted is trusted: the candidate's patch is the canonical diff
DERIVED from the authenticated blobs, and admission is decided only by
``check_candidate`` against the frozen universe.

Durability:

* ``ContentStore`` is content-addressed; an object is written once (temporary
  file, fsync, rename) and an existing object with different bytes is an
  integrity failure - an accepted object is never rewritten;
* ``Journal`` is an append-only, fsynced JSONL checkpoint; a restarted run skips
  every candidate already journaled;
* ``ApiClient`` rate-limits, honours GitHub's rate-limit and Retry-After
  headers, retries transient failures with exponential backoff, does NOT follow
  redirects silently (a redirect is recorded as a rename), and classifies every
  failure into a fixed category.

No model is called.  No validation, consumed-test, confirmation or sealed data
is opened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol
import urllib.error
import urllib.request

from harness.repository_isolation import API, ApiResponse, git_object_id

USER_AGENT = "oneiros-repository-native-acquisition/1 (read-only research)"
FAILURE_CATEGORIES = (
    "network_error", "http_error", "not_found", "rate_limited_exhausted", "redirected",
    "corrupt_response", "hash_mismatch", "git_fetch_failed", "git_object_missing",
    "incomplete_tree_evidence", "integrity_violation",
)


class AcquisitionFailure(RuntimeError):
    def __init__(self, category: str, detail: str):
        assert category in FAILURE_CATEGORIES, category
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


class IntegrityViolation(AcquisitionFailure):
    def __init__(self, detail: str):
        super().__init__("integrity_violation", detail)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fsync_directory(path: Path) -> None:
    try:
        handle = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def _write_once(path: Path, data: bytes) -> None:
    """Create ``path`` with ``data`` durably; an existing file must be identical."""
    if path.exists():
        if path.read_bytes() != data:
            raise IntegrityViolation(f"refusing to rewrite {path.name} with different bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


# --- content-addressed store ----------------------------------------------------------

class ContentStore:
    """Immutable, content-addressed storage for raw bytes and git objects."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _raw_path(self, digest: str) -> Path:
        return self.root / "raw" / digest[:2] / digest

    def _git_path(self, oid: str) -> Path:
        return self.root / "git" / oid[:2] / oid

    def put_raw(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        _write_once(self._raw_path(digest), data)
        return digest

    def get_raw(self, digest: str) -> bytes:
        data = self._raw_path(digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise IntegrityViolation(f"stored raw object {digest} is corrupt")
        return data

    def put_git(self, kind: str, body: bytes, expected_oid: str | None = None) -> str:
        oid = git_object_id(kind, body)
        if expected_oid is not None and oid != expected_oid:
            raise AcquisitionFailure("hash_mismatch",
                                     f"{kind} bytes hash to {oid}, not {expected_oid}")
        _write_once(self._git_path(oid), kind.encode("ascii") + b"\0" + body)
        return oid

    def get_git(self, oid: str) -> tuple[str, bytes]:
        data = self._git_path(oid).read_bytes()
        kind, _, body = data.partition(b"\0")
        if git_object_id(kind.decode("ascii"), body) != oid:
            raise IntegrityViolation(f"stored git object {oid} is corrupt")
        return kind.decode("ascii"), body

    def put_json(self, value: Any) -> str:
        return self.put_raw((json.dumps(value, sort_keys=True, indent=1) + "\n").encode("utf-8"))

    def usage_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())


class Journal:
    """Append-only, fsynced JSONL checkpoint keyed by candidate ID."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # a torn final line from a crash is ignored
                self._entries[entry["key"]] = entry

    def done(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries.values())

    def record(self, key: str, value: Mapping[str, Any]) -> None:
        if key in self._entries:
            raise IntegrityViolation(f"journal entry {key} already recorded")
        entry = {"key": key, "recorded_utc": utc_now(), **value}
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._entries[key] = entry


# --- HTTP ------------------------------------------------------------------------

@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    url: str


class Transport(Protocol):
    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class UrllibTransport:
    """Plain HTTPS GETs; redirects are returned, never followed."""

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.opener = urllib.request.build_opener(_NoRedirect)

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return HttpResponse(response.status, {k.lower(): v for k, v in
                                                      response.headers.items()},
                                    response.read(), url)
        except urllib.error.HTTPError as error:
            return HttpResponse(error.code, {k.lower(): v for k, v in error.headers.items()},
                                error.read() or b"", url)


@dataclass
class RateLimiter:
    """Minimum spacing between calls plus GitHub rate-limit awareness."""
    min_interval: float = 1.0
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    _last: float = field(default=0.0, repr=False)
    waited_seconds: float = 0.0

    def before_call(self) -> None:
        delay = self._last + self.min_interval - self.clock()
        if delay > 0:
            self._wait(delay)
        self._last = self.clock()

    def _wait(self, seconds: float) -> None:
        self.waited_seconds += seconds
        self.sleep(seconds)

    def wait_for_reset(self, headers: Mapping[str, str], attempt: int) -> float:
        if "retry-after" in headers:
            try:
                seconds = float(headers["retry-after"])
            except ValueError:
                seconds = 60.0
        elif headers.get("x-ratelimit-remaining") == "0" and "x-ratelimit-reset" in headers:
            seconds = max(0.0, float(headers["x-ratelimit-reset"]) - self.clock()) + 2
        else:
            seconds = min(900.0, 60.0 * 2 ** attempt)
        self._wait(seconds)
        return seconds


class ApiClient:
    """Read-only GitHub REST GETs with rate limiting, retries and ETags."""

    def __init__(self, transport: Transport, limiter: RateLimiter | None = None,
                 token: str | None = None, max_attempts: int = 6,
                 backoff: Callable[[int], float] = lambda attempt: min(300.0, 5.0 * 2 ** attempt)):
        self.transport = transport
        self.limiter = limiter or RateLimiter()
        self.token = token
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.calls = 0
        self.retries = 0
        #: Wall time spent inside HTTP requests, excluding rate-limit and backoff waits.
        self.api_seconds = 0.0

    def get(self, url: str) -> tuple[ApiResponse, HttpResponse]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT,
                   "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last = "no attempt"
        for attempt in range(self.max_attempts):
            self.limiter.before_call()
            self.calls += 1
            began = time.monotonic()
            try:
                response = self.transport.get(url, headers)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                self.api_seconds += time.monotonic() - began
                last = f"network: {exc}"
                self.retries += 1
                self.limiter._wait(self.backoff(attempt))
                continue
            self.api_seconds += time.monotonic() - began
            if response.status in (301, 302, 307, 308):
                raise AcquisitionFailure("redirected", response.headers.get("location", ""))
            if response.status == 404:
                raise AcquisitionFailure("not_found", url)
            if response.status in (403, 429) and (
                    response.headers.get("x-ratelimit-remaining") == "0"
                    or "retry-after" in response.headers or response.status == 429):
                last = f"rate limited ({response.status})"
                self.retries += 1
                self.limiter.wait_for_reset(response.headers, attempt)
                continue
            if response.status >= 500:
                last = f"server error {response.status}"
                self.retries += 1
                self.limiter._wait(self.backoff(attempt))
                continue
            if response.status != 200:
                raise AcquisitionFailure("http_error", f"{response.status} for {url}")
            try:
                value = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise AcquisitionFailure("corrupt_response", f"{url}: {exc}") from exc
            if not isinstance(value, dict):
                raise AcquisitionFailure("corrupt_response", f"{url}: not a JSON object")
            return ApiResponse(url=url, body=response.body,
                               sha256=hashlib.sha256(response.body).hexdigest(),
                               retrieved_utc=utc_now(),
                               etag=response.headers.get("etag", "")), response
        category = "rate_limited_exhausted" if "rate limited" in last else "network_error"
        raise AcquisitionFailure(category, f"{url}: {last}")


def resolve_repository(client: ApiClient, repository: str) -> tuple[ApiResponse, dict, list]:
    """Canonical repository metadata; a redirect is followed ONLY by re-querying
    the canonical name, and the rename is recorded."""
    renames: list[dict[str, str]] = []
    name = repository.lower()
    for _ in range(3):
        try:
            response, _ = client.get(f"{API}{name}")
        except AcquisitionFailure as failure:
            if failure.category != "redirected":
                raise
            target = failure.detail
            match = re.search(r"/repositories/(\d+)", target)
            if match:  # GitHub redirects renamed repositories to /repositories/<id>
                meta, _ = client.get(f"https://api.github.com/repositories/{match.group(1)}")
                canonical = str(json.loads(meta.body)["full_name"]).lower()
            else:
                canonical = target.rstrip("/").split("/repos/")[-1].lower()
            renames.append({"queried": name, "redirected_to": canonical})
            name = canonical
            continue
        value = json.loads(response.body)
        full = str(value.get("full_name", "")).lower()
        if full != name:
            renames.append({"queried": name, "returned": full})
            name = full
            continue
        return response, value, renames
    raise AcquisitionFailure("redirected", f"{repository} did not settle: {renames}")


# --- git objects ------------------------------------------------------------------

class GitObjectSource(Protocol):
    def read(self, oid: str) -> tuple[str, bytes] | None: ...


class LocalGitRepository:
    """A blob-less bare clone; raw objects are read with ``git cat-file``.

    Missing blobs are fetched lazily from the promisor remote.  Every returned
    object is re-hashed to its ID before it is trusted.
    """

    def __init__(self, path: Path, url: str, git: str = "git"):
        self.path = Path(path)
        self.url = url
        self.git = git

    def _run(self, *args: str, timeout: int = 1800, input_bytes: bytes | None = None) -> bytes:
        result = subprocess.run([self.git, "--git-dir", str(self.path), *args],
                                input=input_bytes, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            raise AcquisitionFailure("git_fetch_failed", (result.stderr or b"").decode(
                "utf-8", "replace")[-400:])
        return result.stdout

    def fetch_since(self, since: str, branch: str) -> None:
        if not (self.path / "HEAD").exists():
            self.path.mkdir(parents=True, exist_ok=True)
            subprocess.run([self.git, "init", "--bare", "-q", str(self.path)], check=True,
                           capture_output=True)
            self._run("remote", "add", "origin", self.url)
            self._run("config", "remote.origin.promisor", "true")
            self._run("config", "remote.origin.partialclonefilter", "blob:none")
        self._run("fetch", "-q", "--filter=blob:none", f"--shallow-since={since}", "origin",
                  f"+refs/heads/{branch}:refs/heads/{branch}")

    def first_parent_commits(self, branch: str, since: str) -> list[dict[str, Any]]:
        output = self._run("log", "--first-parent", f"--since={since}",
                           "--format=%H%x00%P%x00%ct%x00%B%x1e", branch)
        commits = []
        for chunk in output.decode("utf-8", "replace").split("\x1e"):
            parts = chunk.strip("\n").split("\x00")
            if len(parts) < 4 or not parts[0]:
                continue
            commits.append({"oid": parts[0].strip(), "parents": parts[1].split(),
                            "committer_epoch": int(parts[2]), "message": parts[3]})
        return commits

    def read(self, oid: str) -> tuple[str, bytes] | None:
        try:
            kind = self._run("cat-file", "-t", oid, timeout=600).decode().strip()
            body = self._run("cat-file", kind, oid, timeout=600)
        except AcquisitionFailure:
            return None
        if git_object_id(kind, body) != oid:
            raise AcquisitionFailure("hash_mismatch", f"git returned bytes not hashing to {oid}")
        return kind, body


class RecordingObjects(Mapping[str, tuple[str, bytes]]):
    """A lazy object map that records every object actually read.

    Handing it to the isolation module's tree walks yields, as ``accessed``,
    exactly the tree/blob/commit objects the proofs depend on - which is then
    the stored evidence.
    """

    def __init__(self, source: GitObjectSource):
        self.source = source
        self.accessed: dict[str, tuple[str, bytes]] = {}
        self.missing: set[str] = set()

    def __getitem__(self, oid: str) -> tuple[str, bytes]:
        if oid in self.accessed:
            return self.accessed[oid]
        found = self.source.read(oid)
        if found is None:
            self.missing.add(oid)
            raise KeyError(oid)
        if git_object_id(found[0], found[1]) != oid:
            raise AcquisitionFailure("hash_mismatch", f"object bytes do not hash to {oid}")
        self.accessed[oid] = found
        return found

    def get(self, oid, default=None):
        try:
            return self[oid]
        except KeyError:
            return default

    def __iter__(self) -> Iterator[str]:
        return iter(self.accessed)

    def __len__(self) -> int:
        return len(self.accessed)
