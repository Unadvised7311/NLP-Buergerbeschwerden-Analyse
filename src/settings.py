"""CLI-, Umgebungs- und JSON-Konfiguration zusammenführen."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SAMPLE_SIZE = 5000


def _merge_language(cli_lang: str | None, env_val: str | None, file_val: Any) -> str:
    raw = cli_lang
    if raw is None or raw.strip() == "":
        raw = env_val
    if raw is None or raw.strip() == "":
        raw = file_val if isinstance(file_val, str) else None
    if raw is None or str(raw).strip() == "":
        return "en"
    l = str(raw).strip().lower()
    if l in ("de", "deutsch", "german", "ger"):
        return "de"
    if l in ("en", "englisch", "english", "eng"):
        return "en"
    raise ValueError(f"Unbekannte Sprache {raw!r}. Erlaubt: de, en.")
DEFAULT_CHUNK_SIZE = 100_000


@dataclass(frozen=True)
class AnalysisSettings:
    """Parameter für Einlesen, Stichprobe und Vorverarbeitung."""

    data_path_hint: str | None
    text_column: str | None
    sample_size: int
    encoding: str | None
    csv_separator: str | None
    chunk_size: int
    language: str
    config_loaded_from: str | None


def _read_json_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config muss ein JSON-Objekt sein: {path}")
    return data


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Umgebungsvariable {name} muss eine Ganzzahl sein, nicht {raw!r}") from None


def _normalize_sep(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    if s in ("\\t", "tab"):
        return "\t"
    return s


def merge_analysis_settings(
    *,
    config_file_cli: str | None,
    cli_data: str | None,
    cli_text_column: str | None,
    cli_sample_size: int | None,
    cli_encoding: str | None,
    cli_csv_sep: str | None,
    cli_chunk_size: int | None,
    cli_language: str | None,
) -> AnalysisSettings:
    file_data: dict[str, Any] = {}
    config_loaded_from: str | None = None

    if config_file_cli:
        p = Path(config_file_cli).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_file_cli}")
        file_data = _read_json_config(p)
        config_loaded_from = str(p.resolve())
    else:
        default_cfg = Path("analysis.config.json")
        if default_cfg.is_file():
            file_data = _read_json_config(default_cfg)
            config_loaded_from = str(default_cfg.resolve())

    def pick_str(cli_val: str | None, env_name: str, file_key: str) -> str | None:
        if cli_val is not None and cli_val.strip() != "":
            return cli_val.strip()
        env_v = os.environ.get(env_name)
        if env_v is not None and env_v.strip() != "":
            return env_v.strip()
        fv = file_data.get(file_key)
        if isinstance(fv, str) and fv.strip() != "":
            return fv.strip()
        return None

    data_hint = pick_str(cli_data, "NLP_DATA_PATH", "data_path")
    if data_hint is None:
        legacy = os.environ.get("NLP_COMPLAINT_CSV")
        if legacy is not None and legacy.strip() != "":
            data_hint = legacy.strip()

    text_column = pick_str(cli_text_column, "NLP_TEXT_COLUMN", "text_column")

    sample_size = cli_sample_size
    if sample_size is None:
        sample_size = _env_int("NLP_SAMPLE_SIZE")
    if sample_size is None:
        sf = file_data.get("sample_size")
        if isinstance(sf, int) and sf > 0:
            sample_size = sf
    if sample_size is None:
        sample_size = DEFAULT_SAMPLE_SIZE
    if sample_size < 1:
        raise ValueError("sample_size muss mindestens 1 sein.")

    encoding = pick_str(cli_encoding, "NLP_ENCODING", "encoding")

    sep_raw = pick_str(cli_csv_sep, "NLP_CSV_SEP", "csv_separator")
    csv_separator = _normalize_sep(sep_raw)

    chunk_size = cli_chunk_size
    if chunk_size is None:
        chunk_size = _env_int("NLP_CHUNK_SIZE")
    if chunk_size is None:
        cf = file_data.get("chunk_size")
        if isinstance(cf, int) and cf > 0:
            chunk_size = cf
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    file_lang = file_data.get("language")
    file_lang_str = file_lang if isinstance(file_lang, str) else None
    language = _merge_language(cli_language, os.environ.get("NLP_LANGUAGE"), file_lang_str)

    return AnalysisSettings(
        data_path_hint=data_hint,
        text_column=text_column,
        sample_size=sample_size,
        encoding=encoding,
        csv_separator=csv_separator,
        chunk_size=chunk_size,
        language=language,
        config_loaded_from=config_loaded_from,
    )
