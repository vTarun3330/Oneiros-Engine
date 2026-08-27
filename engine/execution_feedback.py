"""Reference-free execution feedback for iterative test generation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from harness.safe_execution import execute_assertions


def collect_execution_feedback(
    candidates: Sequence[str], code_under_test: str
) -> List[Dict[str, Any]]:
    """Run candidates against only the user-visible implementation."""
    if not candidates:
        return []
    rows = execute_assertions(candidates, code_under_test)
    feedback = []
    for rank, row in enumerate(rows, 1):
        status = str(row.get("status", "unknown"))
        error = str(row.get("error", "")).replace("\n", " ")[:240]
        feedback.append({
            "attempt": rank,
            "test": str(row.get("test", ""))[:500],
            "status": status,
            "error": error,
        })
    return feedback


def build_feedback_prompt(
    feedback: Sequence[Mapping[str, Any]],
    *,
    require_novel_shape: bool = False,
) -> str:
    """Format bounded execution observations for the next inference round."""
    if not feedback and not require_novel_shape:
        return ""
    lines = [
        "Execution feedback from earlier attempts on the shown code under test follows.",
        "Do not repeat an earlier assertion. Repair runtime-invalid attempts and explore a new edge case.",
    ]
    if require_novel_shape:
        lines.append(
            "Prefer a different argument structure or boundary shape from the earlier attempts."
        )
    for item in feedback[-8:]:
        error = f"; error={item.get('error')}" if item.get("error") else ""
        lines.append(
            f"- attempt {item.get('attempt')}: {item.get('test')} "
            f"-> {item.get('status', 'unknown')}{error}"
        )
    lines.append("Return ONE new Python assert statement only.")
    return "\n".join(lines)
