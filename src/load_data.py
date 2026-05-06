"""CSV-Einlesen: Kodierung/Trennzeichen, Chunks, konfigurierbare Textspalte."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def resolve_data_path(filepath_hint: str | None) -> str:
    """
    Ermittelt die zu öffnende CSV.

    - Wenn ``filepath_hint`` gesetzt ist: genau diese Datei (muss existieren).
    - Sonst automatisch: ``NLP_DATA_PATH`` / ``NLP_COMPLAINT_CSV``, ``data/rows.csv``,
      ``data/sample_complaints.csv``.
    """
    if filepath_hint:
        p = Path(filepath_hint).expanduser().resolve()
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"Datenpfad nicht gefunden oder keine Datei: {filepath_hint}")

    for env_key in ("NLP_DATA_PATH", "NLP_COMPLAINT_CSV"):
        env_path = os.environ.get(env_key)
        if env_path:
            ep = Path(env_path).expanduser().resolve()
            if ep.is_file():
                log.info("Datenquelle über %s: %s", env_key, ep)
                return str(ep)
            raise FileNotFoundError(f"{env_key} zeigt auf keine Datei: {env_path}")

    primary = Path("data/rows.csv")
    if primary.is_file():
        log.info("Datenquelle (Standard): %s", primary.resolve())
        return str(primary.resolve())

    fallback = Path("data/sample_complaints.csv")
    if fallback.is_file():
        log.warning("data/rows.csv fehlt — verwende Fallback %s", fallback)
        return str(fallback.resolve())

    raise FileNotFoundError(
        "Keine CSV gefunden. Legen Sie die Projektdatei nach data/rows.csv, "
        "setzen Sie NLP_DATA_PATH, oder nutzen Sie --data / einen Eintrag data_path in analysis.config.json."
    )


def _first_line_for_sniff(path: str, encoding: str) -> str:
    with open(path, encoding=encoding, errors="replace") as f:
        return f.readline()


def sniff_separator(path: str, encoding: str) -> str:
    """Heuristik anhand der ersten Zeile (bei komplexen Anführungszeichen ggf. --csv-sep setzen)."""
    line = _first_line_for_sniff(path, encoding)
    best_sep = ","
    best_count = -1
    for cand in (",", ";", "\t", "|"):
        n = line.count(cand)
        if n > best_count:
            best_count = n
            best_sep = cand
    return best_sep if best_count > 0 else ","


def resolve_encoding_and_separator(path: str, encoding_hint: str | None, sep_hint: str | None) -> tuple[str, str]:
    """Wählt Kodierung und Trennzeichen; bei ``encoding_hint`` ``auto`` oder leer wird geraten."""
    auto_enc = encoding_hint is None or encoding_hint.strip().lower() in ("", "auto")

    working_enc = "utf-8"
    if not auto_enc:
        working_enc = encoding_hint.strip()

    sep = sep_hint if sep_hint else sniff_separator(path, working_enc)

    if auto_enc:
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                pd.read_csv(path, nrows=5, encoding=enc, sep=sep, on_bad_lines="skip")
                log.info("CSV-Kodierung automatisch: %s, Trennzeichen: %r", enc, sep)
                return enc, sep
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
                continue
        log.warning("Kodierung nicht sicher erkannt — verwende utf-8 mit Fehlerersetzung.")
        return "utf-8", sep

    try:
        pd.read_csv(path, nrows=5, encoding=working_enc, sep=sep, on_bad_lines="skip")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Kodierung {working_enc!r} passt nicht zu {path}. Versuchen Sie --encoding auto oder latin-1/cp1252."
        ) from e
    log.info("CSV-Kodierung: %s, Trennzeichen: %r", working_enc, sep)
    return working_enc, sep


def _detect_text_column(columns: list[str]) -> str:
    normalized = {c.strip(): c for c in columns}
    lower_map = {c.lower(): c for c in columns}

    for name in (
        "Consumer complaint narrative",
        "consumer_complaint_narrative",
        "Consumer Complaint Narrative",
    ):
        if name in normalized:
            return normalized[name]
        ln = name.lower()
        if ln in lower_map:
            return lower_map[ln]

    for key in ("beschwerdetext", "beschreibung", "beschwerde", "text", "narrative", "kommentar", "inhalt"):
        if key in lower_map:
            return lower_map[key]

    preview = ", ".join(repr(c) for c in columns[:30])
    raise KeyError(
        "Keine Textspalte automatisch erkannt. Verfügbare Spalten (Auszug): "
        f"{preview}. Nutzen Sie --text-column oder text_column in der Config."
    )


def _resolve_text_column(columns: list[str], explicit: str | None) -> str:
    if explicit:
        ex = explicit.strip()
        if ex in columns:
            return ex
        lower_map = {c.lower(): c for c in columns}
        if ex.lower() in lower_map:
            return lower_map[ex.lower()]
        preview = ", ".join(repr(c) for c in columns[:40])
        raise KeyError(f"Spalte {ex!r} nicht gefunden. Verfügbare Spalten (Auszug): {preview}")
    return _detect_text_column(columns)


def load_complaints(
    *,
    path: str,
    text_column: str | None = None,
    sample_size: int = 5000,
    encoding: str | None = None,
    csv_separator: str | None = None,
    chunk_size: int = 100_000,
) -> tuple[pd.Series, dict]:
    """
    Liest bis zu ``sample_size`` **nicht-leere** Texte aus der gewählten Spalte (Chunk-Reading).

    Rückgabe: ``(Serie, Metadaten-Dict)`` für Reporting / JSON-Embed.
    """
    encoding_resolved, sep_resolved = resolve_encoding_and_separator(path, encoding, csv_separator)

    header = pd.read_csv(path, nrows=0, encoding=encoding_resolved, sep=sep_resolved)
    col = _resolve_text_column(list(header.columns), text_column)

    collected: list[str] = []
    chunk_iter = pd.read_csv(
        path,
        usecols=[col],
        chunksize=chunk_size,
        encoding=encoding_resolved,
        encoding_errors="replace",
        sep=sep_resolved,
        low_memory=False,
        on_bad_lines="skip",
    )

    for chunk in chunk_iter:
        s = chunk[col].dropna().astype(str).str.strip()
        s = s[(s != "") & (s.str.lower() != "nan")]
        collected.extend(s.tolist())
        if len(collected) >= sample_size:
            break

    if not collected:
        raise ValueError(f"In {path!r} keine nicht-leeren Einträge in Spalte {col!r}.")

    used_n = min(len(collected), sample_size)
    if len(collected) < sample_size:
        log.warning(
            "Nur %s nicht-leere Texte verfügbar (angefordert: %s). Verwende alle verfügbaren.",
            len(collected),
            sample_size,
        )

    meta = {
        "dateipfad": str(Path(path).resolve()),
        "textspalte": col,
        "stichprobe_geplant": sample_size,
        "stichprobe_effektiv": used_n,
        "encoding": encoding_resolved,
        "csv_trennzeichen": sep_resolved,
    }
    return pd.Series(collected[:sample_size]), meta
