"""spaCy-Vorverarbeitung für DE/EN: Lemma, Filter, Domänen-Stoppwörter."""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import FrozenSet

import spacy

log = logging.getLogger(__name__)

_SPACY_CACHE: dict[str, spacy.language.Language] = {}

MODEL_FOR_LANG: dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
}

CUSTOM_STOPWORDS: dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {"dear", "complaint", "bank", "service", "please", "xxxx", "xxxxxxxx"}
    ),
    "de": frozenset(
        {
            "beschwerde",
            "beschwerden",
            "geehrte",
            "geehrter",
            "sehr",
            "hallo",
            "damen",
            "herren",
            "mitteilung",
            "stadt",
            "verwaltung",
            "bitte",
            "datum",
            "betreff",
            "xxxx",
            "xxxxxxxx",
        }
    ),
}


def normalize_language(lang: str | None) -> str:
    """Gibt ``de`` oder ``en`` zurück."""
    if lang is None or str(lang).strip() == "":
        return "en"
    l = str(lang).strip().lower()
    if l in ("de", "deutsch", "german", "ger"):
        return "de"
    if l in ("en", "englisch", "english", "eng"):
        return "en"
    raise ValueError(f"Unbekannte Sprache {lang!r}. Erlaubt: de, en.")


def _letters_only_nfkc(lemma_lower: str) -> str:
    s = unicodedata.normalize("NFKC", lemma_lower)
    return "".join(ch for ch in s if unicodedata.category(ch).startswith("L"))


def get_nlp(language: str) -> spacy.language.Language:
    """Lädt (und cached) das passende spaCy-Modell."""
    lang = normalize_language(language)
    if lang not in _SPACY_CACHE:
        model_name = MODEL_FOR_LANG[lang]
        try:
            _SPACY_CACHE[lang] = spacy.load(model_name, disable=["parser", "ner"])
        except OSError as e:
            raise OSError(
                f"spaCy-Modell '{model_name}' ist nicht installiert. "
                f"Installieren Sie es mit: python -m spacy download {model_name}"
            ) from e
        log.debug("spaCy-Modell geladen: %s", model_name)
    return _SPACY_CACHE[lang]


def preprocess_pipeline(texts, language: str = "en"):
    """Rohtexte für BoW aufbereiten; language de/en."""
    lang = normalize_language(language)
    nlp = get_nlp(lang)
    custom = CUSTOM_STOPWORDS[lang]

    processed_texts: list[str] = []
    for doc in nlp.pipe(texts, batch_size=500):
        tokens: list[str] = []
        for token in doc:
            if token.is_stop or token.is_punct or token.like_num or token.like_url:
                continue
            lemma_raw = (token.lemma_ or token.norm_ or token.text).strip()
            if not lemma_raw:
                continue
            t_low = lemma_raw.lower()
            clean_t = _letters_only_nfkc(t_low)
            if len(clean_t) <= 2 or clean_t in custom:
                continue
            if re.fullmatch(r"x+", clean_t):
                continue
            tokens.append(clean_t)
        processed_texts.append(" ".join(tokens))
    return processed_texts
