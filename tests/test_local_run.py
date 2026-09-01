"""Synthetic filesystem checks only: never inspect project datasets."""
import pytest
from utils.local_run import local_run_paths


def test_run_outputs_are_isolated_without_creating_them(tmp_path):
    adapter, results = local_run_paths(tmp_path, 'local_sft_42')
    assert adapter == tmp_path / 'checkpoints' / 'local_sft_42'
    assert results == tmp_path / 'results' / 'local_sft_42'
    assert not adapter.exists() and not results.exists()


@pytest.mark.parametrize('name', ['', '../old', 'a/b', 'a\\b', 'C:old', 'CON', 'lpt1', '.hidden'])
def test_invalid_names_cannot_escape_or_alias_outputs(tmp_path, name):
    with pytest.raises(ValueError):
        local_run_paths(tmp_path, name)


@pytest.mark.parametrize('directory', ['checkpoints', 'results'])
def test_fresh_refuses_existing_run_without_changing_it(tmp_path, directory):
    existing = tmp_path / directory / 'old_run'
    existing.mkdir(parents=True)
    marker = existing / 'keep.txt'
    marker.write_text('preserve')
    with pytest.raises(ValueError, match='Refusing'):
        local_run_paths(tmp_path, 'old_run', fresh=True)
    assert marker.read_text() == 'preserve'
    assert local_run_paths(tmp_path, 'old_run')[0].name == 'old_run'
