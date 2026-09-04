"""Defect-family classification for verified real-repository records.

Every repository record in the V4.1 corpus carries the single generic mutation
family ``real_repository_defect``.  That label says where a record came from,
not what is wrong with it, so it cannot support per-family reporting or
family-balanced sampling.

This module assigns a real defect taxonomy using only *permitted offline
supervision evidence*:

* the buggy revision and the fixed revision of the localized region,
* the official failing test's traceback tail,
* patch metadata (which source paths the gold patch touched),
* the official test selector.

Fixed-side text and gold-test bodies are supervision/evaluation metadata.  They
are read HERE, during corpus construction, and are never placed in a model
prompt.  Classification output is corpus metadata only.

The test framework (pytest, unittest) is execution metadata and is deliberately
NOT a defect family.
"""
from __future__ import annotations

import ast
import difflib
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


DEFECT_TAXONOMY_VERSION = "oneiros_repository_defect_taxonomy_v1"

#: The frozen family vocabulary.  ``other_verified_repository_defect`` is the
#: explicit fallback: a record that no rule fires on keeps its verified status
#: and is reported as unclassified rather than forced into a nearby family.
DEFECT_FAMILIES = (
    "boundary_and_control_flow",
    "arithmetic_and_numerical",
    "input_validation",
    "state_mutation",
    "parsing_and_serialization",
    "exception_handling",
    "API_contract",
    "indexing_and_data_structures",
    "configuration_and_integration",
    "filesystem_and_resource_handling",
    "asynchronous_and_concurrency",
    "dependency_interaction",
    "output_formatting",
    "type_and_schema_handling",
    "other_verified_repository_defect",
)
FALLBACK_FAMILY = "other_verified_repository_defect"

#: Confidence is a statement about the *evidence*, not about the model.
#: ``high``   - a decisive structural signal (added guard, changed operator).
#: ``medium`` - a converging but weaker signal (traceback type, path hints).
#: ``low``    - only diffuse lexical evidence supported the choice.
#: ``none``   - nothing fired; the record is the explicit fallback family.
CONFIDENCE_LEVELS = ("none", "low", "medium", "high")


@dataclass(frozen=True)
class DefectClassification:
    taxonomy_version: str
    primary_bug_family: str
    secondary_bug_tags: tuple[str, ...]
    classification_method: str
    classification_confidence: str
    evidence: tuple[str, ...]
    family_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["secondary_bug_tags"] = list(self.secondary_bug_tags)
        payload["evidence"] = list(self.evidence)
        return payload


# --------------------------------------------------------------------------
# Evidence extraction
# --------------------------------------------------------------------------

def diff_lines(buggy: str, fixed: str) -> tuple[list[str], list[str]]:
    """Return (added, removed) payload lines of the buggy->fixed patch."""
    added: list[str] = []
    removed: list[str] = []
    for line in difflib.unified_diff(
        (buggy or "").splitlines(), (fixed or "").splitlines(), lineterm="", n=0,
    ):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return added, removed


def _node_kinds(lines: Sequence[str]) -> set[str]:
    """AST node kinds present in a diff side, parsed leniently.

    Diff fragments are rarely valid standalone Python, so each line is also
    tried on its own with indentation stripped.  This is a best-effort signal
    that supplements the lexical rules; it never has to succeed.
    """
    kinds: set[str] = set()
    joined = "\n".join(line.strip() for line in lines)
    candidates: list[str] = [joined] + [line.strip() for line in lines]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(candidate)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            kinds.add(type(node).__name__)
    return kinds


def traceback_exception(evidence: dict[str, Any] | None) -> str | None:
    """Last exception type named by the official failing test, if recorded."""
    if not isinstance(evidence, dict):
        return None
    tail = evidence.get("buggy_output_tail")
    if not isinstance(tail, str) or not tail.strip():
        return None
    explicit = re.findall(
        r"^E\s+(?:[A-Za-z_][\w.]*\.)?([A-Za-z_]\w*(?:Error|Exception|Warning|Failure))\b",
        tail, re.M,
    )
    if explicit:
        return explicit[-1]
    loose = re.findall(r"\b([A-Z]\w*(?:Error|Exception))\b", tail)
    return loose[-1] if loose else None


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

_GUARD_PATTERN = re.compile(
    r"\b(if|elif)\b.*\b(is\s+(not\s+)?None|not\s+\w|==\s*None|!=\s*None|"
    r"isinstance|hasattr|len\s*\(|in\s+|empty|None\s*(==|!=))",
)
_RAISE_PATTERN = re.compile(r"^\s*raise\b")
_TRY_PATTERN = re.compile(r"^\s*(try|except|finally)\b")
_COMPARISON_CHANGE = re.compile(r"(<=|>=|<|>|==|!=)")
_ARITHMETIC_CHANGE = re.compile(r"(\+|-|\*|/|//|%|\*\*)\s*\d|\bround\b|\babs\b|\bfloat\b|\bint\b\s*\(")
_INDEXING = re.compile(r"\[[^\]]*\]|\.append\b|\.pop\b|\.insert\b|\.extend\b|\.keys\b|\.items\b|\.values\b|\.get\b")
_PARSING = re.compile(
    r"\bjson\b|\byaml\b|\bxml\b|\bparse\w*\b|\bloads?\b|\bdumps?\b|\bencode\b|"
    r"\bdecode\b|\bserial\w*\b|\bdeserial\w*\b|\bregex\b|\bre\.\w+|\bsplit\b|\bstrip\b",
)
_FILESYSTEM = re.compile(
    r"\bopen\s*\(|\bos\.path\b|\bpathlib\b|\bPath\s*\(|\bshutil\b|\bclose\s*\(|"
    r"\bwith\s+open\b|\btempfile\b|\bmkdir\b|\bremove\s*\(|\bunlink\b|\bexists\s*\(",
)
_ASYNC = re.compile(r"\basync\b|\bawait\b|\basyncio\b|\bthread\w*\b|\block\b|\bconcurrent\b|\bqueue\b")
_OUTPUT_FORMAT = re.compile(
    r"%[sdrfx]|\.format\s*\(|\bf\"|\bf'|\brepr\s*\(|\bstr\s*\(|\bprint\s*\(|"
    r"\bdisplay\b|\bmessage\b|\bformat_\w+|\bjoin\s*\(",
)
_TYPE_SCHEMA = re.compile(
    r"\bisinstance\b|\btype\s*\(|\bdtype\b|\bcast\b|\bannotat\w*|\bschema\b|"
    r"\bTypeVar\b|\bOptional\b|\bUnion\b|\bList\[|\bDict\[|\bstr\s*\(|\bint\s*\(|\bbool\s*\(",
)
_CONFIG = re.compile(
    r"\bsetting\w*|\bconfig\w*|\benviron\b|\bgetenv\b|\bdefault\w*|\boption\w*|"
    r"\bflag\w*|\bparam\w*\s*=|\bkwargs\b",
)
_STATE_MUTATION = re.compile(
    r"^\s*self\.\w+\s*(=|\+=|-=|\*=|/=)|\.\w+\s*=\s|\bglobal\b|\bnonlocal\b|"
    r"\.update\s*\(|\.setdefault\s*\(|\bcopy\s*\(|\bdeepcopy\b",
)
_API_CONTRACT = re.compile(
    r"\bdef\s+\w+\s*\(|\breturn\b|\bsuper\s*\(|\b\w+\s*\([^)]*=\s*\w+|"
    r"\bsignature\b|\bdeprecat\w*|\bcallable\b",
)

_EXCEPTION_PRIORS: dict[str, tuple[str, float]] = {
    "TypeError": ("type_and_schema_handling", 2.0),
    "AttributeError": ("API_contract", 1.6),
    "ValueError": ("input_validation", 1.6),
    "IndexError": ("indexing_and_data_structures", 2.4),
    "KeyError": ("indexing_and_data_structures", 2.4),
    "ImportError": ("dependency_interaction", 2.6),
    "ModuleNotFoundError": ("dependency_interaction", 2.6),
    "FileNotFoundError": ("filesystem_and_resource_handling", 2.6),
    "IsADirectoryError": ("filesystem_and_resource_handling", 2.6),
    "PermissionError": ("filesystem_and_resource_handling", 2.4),
    "UnicodeDecodeError": ("parsing_and_serialization", 2.4),
    "UnicodeEncodeError": ("parsing_and_serialization", 2.4),
    "JSONDecodeError": ("parsing_and_serialization", 2.6),
    "TimeoutError": ("asynchronous_and_concurrency", 2.2),
    "RecursionError": ("boundary_and_control_flow", 1.8),
    "ZeroDivisionError": ("arithmetic_and_numerical", 2.6),
    "OverflowError": ("arithmetic_and_numerical", 2.4),
    "NameError": ("dependency_interaction", 1.6),
    "StopIteration": ("boundary_and_control_flow", 1.6),
    "NotImplementedError": ("API_contract", 1.6),
}

_PATH_PRIORS: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"pars|lex|token|serial|json|yaml|xml|encod|decod"), "parsing_and_serialization", 1.0),
    (re.compile(r"valid|check|sanit|clean"), "input_validation", 1.0),
    (re.compile(r"config|setting|option|conf\b"), "configuration_and_integration", 1.0),
    (re.compile(r"async|concurren|thread|worker|queue|pool"), "asynchronous_and_concurrency", 1.0),
    (re.compile(r"format|render|display|print|report|writer"), "output_formatting", 1.0),
    (re.compile(r"\bio\b|file|path|storage|fs\b"), "filesystem_and_resource_handling", 0.8),
)


def _lexical_score(
    added: Sequence[str], removed: Sequence[str],
) -> tuple[dict[str, float], list[str]]:
    """Score families from what the fix added versus what it removed."""
    scores: dict[str, float] = {}
    evidence: list[str] = []

    def bump(family: str, weight: float, note: str) -> None:
        scores[family] = scores.get(family, 0.0) + weight
        evidence.append(note)

    added_text = "\n".join(added)
    removed_text = "\n".join(removed)
    both = added_text + "\n" + removed_text

    # A guard or raise that the fix ADDS is the strongest single signal:
    # the defect was a missing check.
    added_guards = [line for line in added if _GUARD_PATTERN.search(line)]
    removed_guards = [line for line in removed if _GUARD_PATTERN.search(line)]
    if len(added_guards) > len(removed_guards):
        bump("input_validation", 2.6, "fix_adds_guard_condition")
        bump("boundary_and_control_flow", 1.4, "fix_adds_guard_condition")
    elif added_guards or removed_guards:
        bump("boundary_and_control_flow", 1.2, "guard_condition_modified")

    added_raises = [line for line in added if _RAISE_PATTERN.search(line)]
    if added_raises and not any(_RAISE_PATTERN.search(line) for line in removed):
        bump("exception_handling", 2.4, "fix_adds_raise")
    if any(_TRY_PATTERN.search(line) for line in added):
        bump("exception_handling", 2.0, "fix_adds_try_except")

    # An operator that changed on both sides is a boundary/arithmetic defect.
    added_ops = set(_COMPARISON_CHANGE.findall(added_text))
    removed_ops = set(_COMPARISON_CHANGE.findall(removed_text))
    if added_ops and removed_ops and added_ops != removed_ops:
        bump("boundary_and_control_flow", 2.2, "comparison_operator_changed")
    if _ARITHMETIC_CHANGE.search(added_text) and _ARITHMETIC_CHANGE.search(removed_text):
        bump("arithmetic_and_numerical", 1.4, "arithmetic_expression_changed")

    for pattern, family, weight, note in (
        (_INDEXING, "indexing_and_data_structures", 1.2, "collection_access_changed"),
        (_PARSING, "parsing_and_serialization", 1.2, "parsing_call_changed"),
        (_FILESYSTEM, "filesystem_and_resource_handling", 1.4, "filesystem_call_changed"),
        (_ASYNC, "asynchronous_and_concurrency", 1.8, "async_construct_changed"),
        (_OUTPUT_FORMAT, "output_formatting", 1.0, "formatting_expression_changed"),
        (_TYPE_SCHEMA, "type_and_schema_handling", 1.0, "type_handling_changed"),
        (_CONFIG, "configuration_and_integration", 0.8, "configuration_value_changed"),
        (_STATE_MUTATION, "state_mutation", 1.2, "state_write_changed"),
        (_API_CONTRACT, "API_contract", 0.6, "call_or_signature_changed"),
    ):
        if pattern.search(both):
            bump(family, weight, note)

    # Import churn means the fix changed which dependency is used.
    if any(re.match(r"\s*(import|from)\b", line) for line in added + removed):
        bump("dependency_interaction", 1.6, "import_changed")

    return scores, evidence


def classify_repository_defect(
    buggy_code: str,
    fixed_code: str,
    official_test_evidence: dict[str, Any] | None = None,
    patched_source_paths: Iterable[str] | None = None,
    test_selector: str | None = None,
) -> DefectClassification:
    """Assign a defect family from permitted offline supervision evidence.

    All arguments are supervision-side metadata.  Nothing returned here may be
    rendered into a model prompt.
    """
    added, removed = diff_lines(buggy_code or "", fixed_code or "")
    scores, evidence = _lexical_score(added, removed)
    methods: list[str] = []
    if added or removed:
        methods.append("patch_diff")

    kinds_added = _node_kinds(added)
    kinds_removed = _node_kinds(removed)
    if "Try" in kinds_added - kinds_removed or "ExceptHandler" in kinds_added - kinds_removed:
        scores["exception_handling"] = scores.get("exception_handling", 0.0) + 1.6
        evidence.append("ast_exception_node_added")
        methods.append("patch_ast")
    if "If" in kinds_added - kinds_removed:
        scores["boundary_and_control_flow"] = scores.get("boundary_and_control_flow", 0.0) + 1.0
        evidence.append("ast_branch_added")
        methods.append("patch_ast")

    exception_name = traceback_exception(official_test_evidence)
    if exception_name:
        methods.append("traceback")
        prior = _EXCEPTION_PRIORS.get(exception_name)
        if prior:
            family, weight = prior
            scores[family] = scores.get(family, 0.0) + weight
            evidence.append(f"traceback:{exception_name}")
        else:
            evidence.append(f"traceback_uninformative:{exception_name}")

    joined_paths = " ".join(str(item) for item in (patched_source_paths or [])).lower()
    if joined_paths:
        methods.append("patch_paths")
        for pattern, family, weight in _PATH_PRIORS:
            if pattern.search(joined_paths):
                scores[family] = scores.get(family, 0.0) + weight
                evidence.append(f"path_hint:{family}")
    if test_selector:
        selector = str(test_selector).lower()
        for pattern, family, weight in _PATH_PRIORS:
            if pattern.search(selector):
                scores[family] = scores.get(family, 0.0) + weight * 0.5
                evidence.append(f"selector_hint:{family}")

    if not scores:
        return DefectClassification(
            taxonomy_version=DEFECT_TAXONOMY_VERSION,
            primary_bug_family=FALLBACK_FAMILY,
            secondary_bug_tags=(),
            classification_method="+".join(dict.fromkeys(methods)) or "no_evidence",
            classification_confidence="none",
            evidence=tuple(dict.fromkeys(evidence)),
            family_scores={},
        )

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_family, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - runner_up

    if top_score >= 2.4 and margin >= 0.8:
        confidence = "high"
    elif top_score >= 1.4:
        confidence = "medium"
    else:
        confidence = "low"

    secondary = tuple(
        family for family, score in ranked[1:] if score >= 1.0
    )[:3]
    return DefectClassification(
        taxonomy_version=DEFECT_TAXONOMY_VERSION,
        primary_bug_family=top_family,
        secondary_bug_tags=secondary,
        classification_method="+".join(dict.fromkeys(methods)) or "patch_diff",
        classification_confidence=confidence,
        evidence=tuple(dict.fromkeys(evidence)),
        family_scores={family: round(score, 3) for family, score in ranked},
    )
