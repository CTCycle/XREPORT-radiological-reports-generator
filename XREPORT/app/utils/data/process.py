from __future__ import annotations
import os
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from XREPORT.app.constants import TOKENIZERS_PATH


# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:
    def __init__(self, configuration: dict[str, Any], dataframe: pd.DataFrame) -> None:
        # Set the sizes for the train and validation datasets
        self.validation_size = configuration.get("validation_size", 1.0)
        self.seed = configuration.get("split_seed", 42)
        self.train_size = 1.0 - self.validation_size
        self.dataframe = dataframe

        # Compute the sizes of each split
        total_samples = len(dataframe)
        self.train_samples = int(total_samples * self.train_size)
        self.val_samples = int(total_samples * self.validation_size)

    # -------------------------------------------------------------------------
    def split_train_and_validation(self) -> pd.DataFrame:
        dataframe = self.dataframe.sample(frac=1.0, random_state=self.seed).reset_index(
            drop=True
        )
        dataframe.loc[: self.train_samples - 1, "split"] = "train"
        dataframe.loc[
            self.train_samples : self.train_samples + self.val_samples - 1, "split"
        ] = "validation"

        return dataframe


# [TOKENIZER]
###############################################################################
class TextSanitizer:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.max_report_size = configuration.get("max_report_size", 200)
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def sanitize_text(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["text"] = dataset["text"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)

        return dataset


# [TOKENIZER]
###############################################################################
class TokenizerHandler:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.tokenizer_id = configuration.get("tokenizer", None)
        self.max_report_size = configuration.get("max_report_size", 200)
        result = self.get_tokenizer(self.tokenizer_id)
        self.tokenizer, self.vocabulary_size = (
            result if result is not None else (None, 0)
        )
        self.pad_token = (
            self.tokenizer.pad_token_id if self.tokenizer is not None else None
        )

    # -------------------------------------------------------------------------
    def get_tokenizer(
        self, tokenizer_name: str | None = None
    ) -> None | tuple[Any, int]:
        if tokenizer_name is None:
            return

        tokenizer_path = os.path.join(TOKENIZERS_PATH, tokenizer_name)
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=tokenizer_path
        )
        vocabulary_size = len(tokenizer.vocab)

        return tokenizer, vocabulary_size

    # -------------------------------------------------------------------------
    def tokenize_text_corpus(self, data: pd.DataFrame) -> pd.DataFrame:
        # tokenize train and validation text using loaded tokenizer
        true_report_size = self.max_report_size + 1
        text = data["text"].to_list()
        if self.tokenizer is None:
            return data

        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=true_report_size,
            return_tensors="pt",
        )

        # add tokenizer sequences to the source dataframe, now containing the paths,
        # original text and tokenized text
        data["tokens"] = tokens["input_ids"].numpy().tolist()

        return data
