from __future__ import annotations

import os

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ["KERAS_BACKEND"] = "torch"

from keras import ops  # noqa: E402

from server.api import tokenizers as tokenizers_api  # noqa: E402
from server.configurations.tokenizers import TOKENIZER_CATALOGUE  # noqa: E402
from server.models.inference.generator import TextGenerator  # noqa: E402
from server.models.training.layers import PositionalEmbedding  # noqa: E402
from server.models.training.metrics import (  # noqa: E402
    MaskedAccuracy,
    MaskedSparseCategoricalCrossentropy,
)
from server.models.training.processing import (  # noqa: E402
    TokenizerCompatibilityError,
    TokenizerHandler,
    TokenizerUnavailableError,
    is_cached_tokenizer,
    resolve_special_tokens,
)
from server.repositories.serialization.dataset import DatasetRepository  # noqa: E402
from server.services import tokenizers as tokenizer_service  # noqa: E402

###############################################################################
class FakeTokenizer:
    """Minimal stand-in exposing the attributes the decoder integration reads."""

    def __init__(self, ids: dict[str, int | None], vocabulary: dict[str, int]) -> None:
        self.vocabulary = vocabulary
        self._special = ids
        self.cls_token = "[CLS]" if ids.get("cls") is not None else None
        self.sep_token = "[SEP]" if ids.get("sep") is not None else None
        self.bos_token = "<s>" if ids.get("bos") is not None else None
        self.eos_token = "</s>" if ids.get("eos") is not None else None
        self.pad_token = "[PAD]" if ids.get("pad") is not None else None
        self.pad_token_id = ids.get("pad")
        self.decoded: list[list[int]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"[CLS]": self._special.get("cls"), "[SEP]": self._special.get("sep"),
                "<s>": self._special.get("bos"), "</s>": self._special.get("eos")}[token]

    def decode(self, ids: list[int], **_kwargs) -> str:
        self.decoded.append(list(ids))
        lookup = {value: key for key, value in self.vocabulary.items()}
        return " ".join(lookup[item] for item in ids)

    def __len__(self) -> int:
        return len(self.vocabulary)

BERT = {"cls": 101, "sep": 102, "pad": 0}
ROBERTA = {"bos": 0, "eos": 2, "pad": 1}

###############################################################################
def test_special_tokens_follow_cls_sep_or_bos_eos_and_require_all_three() -> None:
    bert = resolve_special_tokens(FakeTokenizer(BERT, {"a": 5}))
    assert (bert.start_token_id, bert.end_token_id, bert.pad_token_id) == (101, 102, 0)
    roberta = resolve_special_tokens(FakeTokenizer(ROBERTA, {"a": 5}))
    assert (roberta.start_token_id, roberta.end_token_id, roberta.pad_token_id) == (0, 2, 1)
    with pytest.raises(TokenizerCompatibilityError, match="start"):
        resolve_special_tokens(FakeTokenizer({"eos": 1, "pad": 0}, {"a": 5}))
    with pytest.raises(TokenizerCompatibilityError, match="padding"):
        resolve_special_tokens(FakeTokenizer({"bos": 0, "eos": 2}, {"a": 5}))

###############################################################################
def test_generator_pads_and_decodes_with_the_tokenizer_ids() -> None:
    tokenizer = FakeTokenizer(ROBERTA, {"<s>": 0, "<pad>": 1, "</s>": 2, "no": 10, "effusion": 11})
    generator = TextGenerator(model=None, configuration={}, max_report_size=6)  # type: ignore[arg-type]
    config = {"start_token_idx": 0, "end_token_idx": 2, "pad_token_idx": 1, "tokenizer": tokenizer}

    sequence = generator._build_sequence_input([0, 10, 11], pad_token_idx=1)
    assert sequence.tolist() == [[0, 10, 11, 1, 1, 1]]
    text = generator.translate_tokens_to_text(sequence, config)

    # Start id 0 and pad id 1 are dropped even though 0 is a real token for RoBERTa.
    assert tokenizer.decoded == [[10, 11]]
    assert text == "no effusion"

###############################################################################
def test_loss_and_accuracy_ignore_the_configured_pad_id() -> None:
    y_true = np.array([[7, 1, 1]])
    probabilities = np.full((1, 3, 10), 0.01, dtype="float32")
    probabilities[0, 0, 7] = 0.91
    probabilities[0, 1:, 3] = 0.91  # wrong on pad positions, which must not count
    loss = float(ops.convert_to_numpy(MaskedSparseCategoricalCrossentropy(pad_token_id=1)(y_true, probabilities)))
    assert loss == pytest.approx(-np.log(0.91), rel=1e-4)
    metric = MaskedAccuracy(pad_token_id=1)
    metric.update_state(y_true, probabilities)
    assert float(ops.convert_to_numpy(metric.result())) == pytest.approx(1.0, abs=1e-4)
    # With the default pad id 0 the same targets are all real tokens.
    default_metric = MaskedAccuracy()
    default_metric.update_state(y_true, probabilities)
    assert float(ops.convert_to_numpy(default_metric.result())) < 0.5

###############################################################################
def test_positional_embedding_masks_the_pad_id_and_defaults_to_zero_for_old_configs() -> None:
    layer = PositionalEmbedding(10, 4, 5, pad_token_id=1)
    inputs = np.array([[3, 4, 1, 1, 1]], dtype="int32")
    assert ops.convert_to_numpy(layer.compute_mask(inputs)).tolist() == [[True, True, False, False, False]]
    config = layer.get_config()
    assert config["pad_token_id"] == 1
    del config["pad_token_id"]
    assert PositionalEmbedding.from_config(config).pad_token_id == 0

###############################################################################
def test_processing_hash_changes_with_tokenizer_revision_and_special_ids() -> None:
    base = {
        "dataset_name": "d", "seed": 1, "sample_size": 1.0, "validation_size": 0.2,
        "vocabulary_size": 30522, "max_report_size": 200, "tokenizer": "bert-base-uncased",
        "tokenizer_revision": "aaa", "pad_token_id": 0, "start_token_id": 101,
        "end_token_id": 102, "split_seed": 42,
    }
    reference = DatasetRepository.generate_hashcode(base)
    assert DatasetRepository.generate_hashcode(dict(base)) == reference
    for key, value in (
        ("tokenizer", "roberta-base"),
        ("tokenizer_revision", "bbb"),
        ("pad_token_id", 1),
        ("split_seed", 7),
    ):
        assert DatasetRepository.generate_hashcode({**base, key: value}) != reference

###############################################################################
def test_check_reports_unavailable_and_incompatible_tokenizers(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str):
        raise TokenizerUnavailableError(f"Tokenizer '{name}' could not be loaded: offline")

    monkeypatch.setattr(tokenizer_service, "load_tokenizer", unavailable)
    missing = tokenizer_service.check_tokenizer("org/missing")
    assert (missing.available, missing.compatible) == (False, False)
    assert "offline" in missing.message

    monkeypatch.setattr(
        tokenizer_service,
        "load_tokenizer",
        lambda _name: FakeTokenizer({"eos": 1, "pad": 0}, {"a": 5}),
    )
    no_start = tokenizer_service.check_tokenizer("t5-small")
    assert (no_start.available, no_start.compatible) == (True, False)
    assert no_start.vocabulary_size == 1 and no_start.start_token_id is None

    monkeypatch.setattr(
        tokenizer_service,
        "load_tokenizer",
        lambda _name: FakeTokenizer(ROBERTA, {"a": 5, "b": 6, "c": 7}),
    )
    ok = tokenizer_service.check_tokenizer("roberta-base")
    assert ok.compatible and (ok.start_token_id, ok.end_token_id, ok.pad_token_id) == (0, 2, 1)

###############################################################################
def test_tokenizer_api_lists_groups_and_rejects_path_like_ids() -> None:
    application = FastAPI()
    application.include_router(tokenizers_api.router, prefix="/api")
    client = TestClient(application)

    listed = client.get("/api/tokenizers").json()["tokenizers"]
    assert len(listed) == len(TOKENIZER_CATALOGUE)
    groups = {item["group"] for item in listed}
    assert {"BERT family", "RoBERTa family", "Biomedical and clinical", "T5 family"} <= groups
    assert not next(item for item in listed if item["tokenizer_id"] == "t5-small")["supports_training"]
    for bad in ("../outside", "a/b/c", "C:\\temp", ""):
        assert client.post("/api/tokenizers/check", json={"tokenizer": bad}).status_code == 422

###############################################################################
@pytest.mark.skipif(
    not is_cached_tokenizer("distilbert-base-uncased"),
    reason="distilbert-base-uncased tokenizer is not cached locally",
)
def test_cached_real_tokenizer_pads_truncates_and_records_special_tokens() -> None:
    handler = TokenizerHandler({"tokenizer": "distilbert-base-uncased", "max_report_size": 20})
    special = handler.special_tokens()
    assert (special.start_token_id, special.end_token_id, special.pad_token_id) == (101, 102, 0)
    assert handler.vocabulary_size == len(handler.tokenizer)

    import pandas as pd

    frame = pd.DataFrame({"text": ["no acute findings", "word " * 100]})
    tokens = handler.tokenize_text_corpus(frame)["tokens"].tolist()
    assert all(len(row) == 21 for row in tokens)
    short = tokens[0]
    end = short.index(102)
    assert short[0] == 101 and end > 1 and set(short[end + 1 :]) == {0}
    assert tokens[1][0] == 101 and tokens[1][-1] == 102
    assert handler.revision is not None
