from scripts.diagnose_execution_supervision_pilot import _format_class


def test_format_classifier_separates_recoverable_markdown_wrappers():
    assert _format_class("") == "empty"
    assert _format_class("**assert f(1) == 2**") == "markdown_bold"
    assert _format_class("```python\nassert f(1) == 2\n```") == "markdown_fence"
    assert _format_class("assert f(1) == 2") == "plain_or_other"
