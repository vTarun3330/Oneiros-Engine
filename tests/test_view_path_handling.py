"""A finished build must not be thrown away by a cosmetic manifest field.

Both corpus builders described their input with
``view_dir.relative_to(ROOT)``, which raises when the caller passes an absolute
path. That call is the LAST statement of the build, so a run executed 667
lineages of kill-matrix verification - about five minutes of sandboxed
execution - and then discarded all of it with a ValueError while writing the
manifest. Twice.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_corpus_inventory import _relative_view as inventory_relative
from scripts.build_multi_mutant_dataset import _relative_view as mutant_relative


def test_a_relative_path_inside_the_repo_stays_relative():
    for helper in (inventory_relative, mutant_relative):
        assert helper(Path("data/corpus/x/development_view")) == (
            "data/corpus/x/development_view"
        )


def test_an_absolute_path_inside_the_repo_is_made_relative():
    """This is the call that crashed: absolute in, and it must not raise."""
    inside = ROOT / "data" / "corpus" / "x" / "development_view"
    for helper in (inventory_relative, mutant_relative):
        assert helper(inside) == "data/corpus/x/development_view"


def test_a_path_outside_the_repo_falls_back_instead_of_raising():
    outside = Path(ROOT.anchor) / "somewhere" / "else"
    for helper in (inventory_relative, mutant_relative):
        result = helper(outside)
        assert result, "a describable path must be returned, not an exception"
        assert "somewhere" in result
