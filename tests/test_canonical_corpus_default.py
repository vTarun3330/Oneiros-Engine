import inspect

from config import CANONICAL_CORPUS_VERSION
from scripts import modal_train
from scripts import train_on_dataset


def test_phase3_entry_points_share_the_canonical_corpus_default():
    assert CANONICAL_CORPUS_VERSION == "v4_1_research_hardened_candidate"
    assert train_on_dataset.CORPUS_VERSION == CANONICAL_CORPUS_VERSION
    assert (
        inspect.signature(modal_train.run_cloud_training.get_raw_f())
        .parameters["corpus_version"]
        .default
        == CANONICAL_CORPUS_VERSION
    )
    assert (
        inspect.signature(modal_train.training_main.info.raw_f)
        .parameters["corpus_version"]
        .default
        == CANONICAL_CORPUS_VERSION
    )
