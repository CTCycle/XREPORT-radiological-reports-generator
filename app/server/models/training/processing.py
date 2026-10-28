from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from server.common.path import TOKENIZERS_DIR

###############################################################################
class TrainValidationSplit:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any], dataframe: pd.DataFrame) -> None:
        self.validation_size = configuration.get("validation_size", 0.2)
        self.seed = configuration.get("split_seed", 42)
        self.train_size = 1.0 - self.validation_size
        self.dataframe = dataframe

        total_samples = len(dataframe)
        self.train_samples = int(total_samples * self.train_size)
        self.val_samples = int(total_samples * self.validation_size)

    # -------------------------------------------------------------------------
    def split_train_and_validation(self) -> pd.DataFrame:
        dataframe = self.dataframe.sample(frac=1.0, random_state=self.seed).reset_index(
            drop=True
        )
        total_samples = len(dataframe)
        if total_samples == 0:
            dataframe["split"] = pd.Series(dtype="string")
            return dataframe

        validation_size = float(self.validation_size)
        validation_size = min(max(validation_size, 0.0), 1.0)
        validation_samples = int(total_samples * validation_size)
        validation_samples = min(max(validation_samples, 0), total_samples)

        self.val_samples = validation_samples
        self.train_samples = total_samples - validation_samples

        dataframe["split"] = "train"
        if validation_samples > 0:
            dataframe.loc[: validation_samples - 1, "split"] = "validation"

        return dataframe

###############################################################################
class TextSanitizer:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.max_report_size = configuration.get("max_report_size", 200)
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def sanitize_text(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["text"] = dataset["text"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        return dataset

###############################################################################
class TokenizerError(RuntimeError):
    """Base error for tokenizer resolution failures that users can act on."""

###############################################################################
class TokenizerUnavailableError(TokenizerError):
    """The tokenizer files cannot be loaded (unknown id, offline, missing library)."""

###############################################################################
class TokenizerCompatibilityError(TokenizerError):
    """The tokenizer cannot drive the report decoder (missing special tokens)."""

###############################################################################
@dataclass(frozen=True)
class SpecialTokens:
    """Ids the decoder needs: sequence start, sequence end and padding."""

    start_token: str
    end_token: str
    pad_token: str
    start_token_id: int
    end_token_id: int
    pad_token_id: int

###############################################################################
def resolve_special_tokens(tokenizer: Any) -> SpecialTokens:
    """Map a tokenizer to decoder special tokens (CLS/SEP or BOS/EOS, plus PAD)."""
    start = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
    end = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
    missing = [
        label
        for label, value in (
            ("start (CLS/BOS)", start),
            ("end (SEP/EOS)", end),
            ("padding", tokenizer.pad_token),
        )
        if value is None
    ]
    if missing:
        raise TokenizerCompatibilityError(
            "Tokenizer defines no " + ", ".join(missing) + " token, which report "
            "generation requires"
        )
    return SpecialTokens(
        start_token=str(start),
        end_token=str(end),
        pad_token=str(tokenizer.pad_token),
        start_token_id=int(tokenizer.convert_tokens_to_ids(start)),
        end_token_id=int(tokenizer.convert_tokens_to_ids(end)),
        pad_token_id=int(tokenizer.pad_token_id),
    )

###############################################################################
def exchange_tokenizer_path(tokenizer_name: str) -> Path:
    return TOKENIZERS_DIR / "_exchange" / tokenizer_name.replace("/", "--")

###############################################################################
def cached_tokenizer_revision(tokenizer_name: str) -> str | None:
    """Commit hash of the cached Hugging Face snapshot for this tokenizer, if any."""
    ref = (
        TOKENIZERS_DIR
        / tokenizer_name
        / f"models--{tokenizer_name.replace('/', '--')}"
        / "refs"
        / "main"
    )
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None

###############################################################################
def is_cached_tokenizer(tokenizer_name: str) -> bool:
    exchange = exchange_tokenizer_path(tokenizer_name)
    return (exchange.is_dir() and any(exchange.iterdir())) or (
        cached_tokenizer_revision(tokenizer_name) is not None
    )

###############################################################################
def load_tokenizer(tokenizer_name: str) -> Any:
    """Load tokenizer assets only (never model weights), preferring local files."""
    from transformers import AutoTokenizer

    # Tokenizers installed from a checkpoint exchange archive load offline.
    exchange_path = exchange_tokenizer_path(tokenizer_name)
    if exchange_path.is_dir() and any(exchange_path.iterdir()):
        return AutoTokenizer.from_pretrained(str(exchange_path), local_files_only=True)

    cache_dir = TOKENIZERS_DIR / tokenizer_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=str(cache_dir), local_files_only=True
        )
    except OSError:
        pass
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=str(cache_dir))
    except Exception as exc:  # noqa: BLE001
        raise TokenizerUnavailableError(
            f"Tokenizer '{tokenizer_name}' could not be loaded: {exc}"
        ) from exc

###############################################################################
class TokenizerHandler:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.tokenizer_id = configuration.get("tokenizer", None)
        self.max_report_size = configuration.get("max_report_size", 200)
        self.tokenizer = (
            load_tokenizer(self.tokenizer_id) if self.tokenizer_id is not None else None
        )
        # `len(tokenizer)` includes added tokens, which `vocab` omits.
        self.vocabulary_size = len(self.tokenizer) if self.tokenizer is not None else 0
        self.pad_token = (
            self.tokenizer.pad_token_id if self.tokenizer is not None else None
        )
        self.revision = (
            cached_tokenizer_revision(self.tokenizer_id)
            if self.tokenizer_id is not None
            else None
        )

    # -------------------------------------------------------------------------
    def special_tokens(self) -> SpecialTokens:
        if self.tokenizer is None:
            raise TokenizerUnavailableError("No tokenizer is configured")
        return resolve_special_tokens(self.tokenizer)

    # -------------------------------------------------------------------------
    def tokenize_text_corpus(self, data: pd.DataFrame) -> pd.DataFrame:
        true_report_size = self.max_report_size + 1
        text = data["text"].to_list()
        if self.tokenizer is None:
            return data

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=true_report_size,
            return_tensors="pt",
        )

        data["tokens"] = tokens["input_ids"].numpy().tolist()

        return data
