"""Curated tokenizer catalogue for dataset processing.

The catalogue is a convenience list. Any Hugging Face tokenizer id that matches
`TOKENIZER_ID_PATTERN` may be used; compatibility is verified by loading the
tokenizer (`check_tokenizer`), not assumed from this table.
"""

from __future__ import annotations

from dataclasses import dataclass

TOKENIZER_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$"
DEFAULT_TOKENIZER = "distilbert-base-uncased"


###############################################################################
@dataclass(frozen=True)
class TokenizerEntry:
    tokenizer_id: str
    label: str
    group: str
    kind: str  # wordpiece | bpe | sentencepiece
    supports_training: bool = True
    note: str | None = None

###############################################################################
_NO_START_TOKEN = (
    "No start token (CLS/BOS): report generation needs one, so it cannot be used "
    "to train the report decoder."
)

TOKENIZER_CATALOGUE: tuple[TokenizerEntry, ...] = (
    TokenizerEntry("distilbert-base-uncased", "DistilBERT base uncased", "BERT family", "wordpiece"),
    TokenizerEntry("distilbert-base-cased", "DistilBERT base cased", "BERT family", "wordpiece"),
    TokenizerEntry("bert-base-uncased", "BERT base uncased", "BERT family", "wordpiece"),
    TokenizerEntry("bert-base-cased", "BERT base cased", "BERT family", "wordpiece"),
    TokenizerEntry("bert-large-uncased", "BERT large uncased", "BERT family", "wordpiece"),
    TokenizerEntry("roberta-base", "RoBERTa base", "RoBERTa family", "bpe"),
    TokenizerEntry("distilroberta-base", "DistilRoBERTa base", "RoBERTa family", "bpe"),
    TokenizerEntry("roberta-large", "RoBERTa large", "RoBERTa family", "bpe"),
    TokenizerEntry("dmis-lab/biobert-v1.1", "BioBERT v1.1", "Biomedical and clinical", "wordpiece"),
    TokenizerEntry("emilyalsentzer/Bio_ClinicalBERT", "Bio_ClinicalBERT", "Biomedical and clinical", "wordpiece"),
    TokenizerEntry(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "BiomedBERT (PubMedBERT) base",
        "Biomedical and clinical",
        "wordpiece",
    ),
    TokenizerEntry("allenai/scibert_scivocab_uncased", "SciBERT uncased", "Biomedical and clinical", "wordpiece"),
    TokenizerEntry("microsoft/deberta-v3-base", "DeBERTa v3 base", "DeBERTa family", "sentencepiece"),
    TokenizerEntry("microsoft/deberta-base", "DeBERTa base", "DeBERTa family", "bpe"),
    TokenizerEntry("microsoft/deberta-large", "DeBERTa large", "DeBERTa family", "bpe"),
    TokenizerEntry("albert-base-v2", "ALBERT base v2", "SentencePiece", "sentencepiece"),
    TokenizerEntry("xlm-roberta-base", "XLM-RoBERTa base", "SentencePiece", "sentencepiece"),
    TokenizerEntry("t5-small", "T5 small", "T5 family", "sentencepiece", False, _NO_START_TOKEN),
    TokenizerEntry("google/flan-t5-base", "Flan-T5 base", "T5 family", "sentencepiece", False, _NO_START_TOKEN),
)


###############################################################################
def catalogue_entry(tokenizer_id: str) -> TokenizerEntry | None:
    return next(
        (item for item in TOKENIZER_CATALOGUE if item.tokenizer_id == tokenizer_id),
        None,
    )
