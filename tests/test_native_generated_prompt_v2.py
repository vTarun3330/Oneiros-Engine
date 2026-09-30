"""Prompt builder v2: deterministic buggy-source-only context (amendment v2.2 section C)."""
from __future__ import annotations

import inspect
import textwrap

from harness import native_generated_test_prompt as prompt

SOURCE = textwrap.dedent('''
    """Module docstring."""
    from __future__ import annotations

    import json
    import os.path
    import re as regex
    from collections import OrderedDict, defaultdict, deque
    from typing import Any, Optional

    try:
        import ujson as fastjson
    except ImportError:
        fastjson = None

    PATTERN = regex.compile(r"[a-z]+")
    UNUSED_CONSTANT = 99
    BIG_TABLE = {%s}
    DEFAULT_SEP = ","


    def register(fn):
        return fn


    def helper_function(value: int, sep: str = DEFAULT_SEP) -> str:
        """Join things."""
        return sep.join([str(value)] * 3)


    def unrelated_function():
        return os.path.join("a", "b")


    class Marker:
        """A marker type used in annotations."""
        def body_should_not_appear(self):
            return "SECRET-BODY-MARKER"


    class Base:
        """Base behaviour."""
        def __init__(self, data: Optional[dict] = None, *, strict: bool = False):
            self.data = data or {}
            self.strict = strict

        def inherited(self, key: str) -> Any:
            """Look a key up."""
            return self.data.get(key, "INHERITED-BODY-MARKER")

        def overridden(self):
            return "BASE-OVERRIDDEN-BODY"


    class Outer(Base):
        # COMMENT-BETWEEN-HEADER-AND-BODY must not appear
        """Outer container.

        More detail that must not appear.
        """
        LIMIT = 10
        TABLE = BIG_TABLE
        SIBLING_ATTR = "unused"

        def sibling(self):
            return "SIBLING-BODY-MARKER"

        def one_liner(self, x): return x + 12345

        def overridden(self):
            return super().overridden()

        class Inner:
            """Inner class."""
            SCALE = 3

            def __init__(self, n: int):
                self.n = n

            def _helper(self, marker: Marker) -> int:
                return self.n * 777

            def unused_inner(self):
                return "UNUSED-INNER-BODY"

            @register
            def target(self, value: "int", marker: Marker = None, *, sep: str = DEFAULT_SEP
                       ) -> Optional[str]:
                """Target docstring."""
                if PATTERN.match(str(value)):
                    return helper_function(value * self.SCALE, sep)
                return str(self._helper(marker)) + str(Outer.LIMIT) + json.dumps(value)

        def uses_base(self, key):
            """Uses inherited helpers."""
            self.one_liner(1)
            return self.inherited(key), super().overridden(), Outer.TABLE


    @dataclass_like
    class Record:
        name: str
        size: int = 0
        CONST = "not a field"

        def describe(self) -> str:
            return f"{self.name}:{self.size}"
''') % ", ".join(f'"k{i}": {i}' for i in range(200))
DTO = {"target_key": "cand:o/r@abc", "repository": "o/r", "buggy_commit": "abc",
       "target_file": "src/pkg/mod.py", "qualname": "Outer.Inner.target"}


def _build(qualname: str, source: str = SOURCE) -> dict:
    return prompt.build_prompt({**DTO, "qualname": qualname}, source)


def test_version_and_determinism():
    assert prompt.BUILDER_VERSION == "oneiros_native_generated_test_prompt_v2"
    first, second = _build("Outer.Inner.target"), _build("Outer.Inner.target")
    assert first == second
    assert first["view_sha256"] and first["prompt_sha256"]


def test_full_target_source_with_decorator_signature_and_docstring_is_retained():
    sealed = _build("Outer.Inner.target")
    source = textwrap.dedent('''
        @register
        def target(self, value: "int", marker: Marker = None, *, sep: str = DEFAULT_SEP
                   ) -> Optional[str]:
            """Target docstring."""
            if PATTERN.match(str(value)):
                return helper_function(value * self.SCALE, sep)
            return str(self._helper(marker)) + str(Outer.LIMIT) + json.dumps(value)''').strip("\n")
    assert sealed["target_source"] == source
    assert source in sealed["prompt"]


def test_nested_class_chain_constructor_and_direct_helpers():
    text = _build("Outer.Inner.target")["prompt"]
    assert "class Outer(Base):\n    \"\"\"Outer container.\"\"\"" in text
    assert "More detail that must not appear." not in text          # short docstrings only
    assert "COMMENT-BETWEEN-HEADER-AND-BODY" not in text             # header lines only
    assert "    class Inner:\n        \"\"\"Inner class.\"\"\"" in text
    assert "def __init__(self, n: int):" in text                     # innermost constructor
    assert "def _helper(self, marker: Marker) -> int:" in text       # direct helper
    assert "SCALE = 3" in text and "LIMIT = 10" in text               # referenced attributes
    assert "# target method (full source below)" in text


def test_irrelevant_siblings_and_all_helper_bodies_are_omitted():
    sealed = _build("Outer.Inner.target")
    text = sealed["prompt"]
    for absent in ("SIBLING-BODY-MARKER", "def sibling", "UNUSED-INNER-BODY", "unused_inner",
                   "SIBLING_ATTR", "self.n * 777", "sep.join", "SECRET-BODY-MARKER",
                   "UNUSED_CONSTANT", "unrelated_function", "import os.path", "OrderedDict",
                   "INHERITED-BODY-MARKER", "class Record"):
        assert absent not in text, absent
    omitted = sealed["omitted"]
    assert omitted["enclosing_class_members_not_referenced"] >= 4
    assert omitted["imports_not_referenced"] >= 4
    assert omitted["helper_bodies_elided"] >= 3


def test_annotation_decorator_and_default_names_are_resolved():
    text = _build("Outer.Inner.target")["prompt"]
    assert "from __future__ import annotations" in text
    assert "import json" in text                                     # used in the body
    assert "from typing import Any, Optional" not in text            # reduced to used aliases
    assert "from typing import Optional" in text                     # return annotation
    assert "def register(fn):" in text                               # decorator name
    assert 'DEFAULT_SEP = ","' in text                               # default value
    assert "class Marker:\n    \"\"\"A marker type used in annotations.\"\"\"" in text
    assert 'PATTERN = regex.compile(r"[a-z]+")' in text and "import re as regex" in text
    assert "def helper_function(value: int, sep: str = DEFAULT_SEP) -> str:" in text


def test_inherited_helpers_super_one_liners_and_oversized_values():
    sealed = _build("Outer.uses_base")
    text = sealed["prompt"]
    assert "class Base:" in text and "def inherited(self, key: str) -> Any:" in text
    assert "def __init__(self, data: Optional[dict] = None, *, strict: bool = False):" in text
    assert "BASE-OVERRIDDEN-BODY" not in text and "def overridden(self):" in text
    assert "def one_liner(self, x):" in text and "12345" not in text  # one-line body elided
    assert "TABLE = BIG_TABLE" in text
    assert "BIG_TABLE = ..." in text and '"k150"' not in text
    assert sealed["omitted"]["oversized_value_elided"] == ["BIG_TABLE"]


def test_decorated_class_without_constructor_shows_its_fields():
    source = SOURCE.replace("@dataclass_like", "@dataclass_like\n")
    text = _build("Record.describe", source)["prompt"]
    assert "name: str" in text and "size: int = 0" in text
    assert 'CONST = "not a field"' not in text


def test_module_function_target_and_try_except_imports():
    source = SOURCE + textwrap.dedent('''
        def dump(value):
            return (fastjson or json).dumps(value)
    ''')
    sealed = _build("dump", source)
    text = sealed["prompt"]
    assert "import ujson as fastjson" in text and "fastjson = None" in text
    assert "import json" in text and "class Outer" not in text
    assert "Enclosing class" not in text


def test_selection_depends_only_on_dto_and_buggy_source():
    params = inspect.signature(prompt.permitted_view).parameters
    assert list(params) == ["dto", "buggy_source"]
    source = inspect.getsource(prompt).split('"""', 2)[2]
    for forbidden in ("open(", "read_text", "fixed", "official", "sidecar", "manifest",
                      "issue", "patch", "verdict", "leak"):
        assert forbidden not in source, forbidden


def test_no_truncation_of_a_long_target():
    body = "\n".join(f"    x{i} = {i}" for i in range(400))
    source = f"def big():\n{body}\n    return x0\n"
    sealed = _build("big", source)
    assert sealed["target_source"] == source.rstrip("\n")
    assert sealed["target_source"] in sealed["prompt"]
