import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt

from src.load_data import load_complaints, resolve_data_path
from src.preprocess import preprocess_pipeline
from src.settings import merge_analysis_settings
from src.topic_modeling import export_topics_to_json, run_topic_modeling
from src.visualize import plot_top_words


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Themenextraktion aus Beschwerde-CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Pfad zu einer JSON-Konfigurationsdatei (siehe analysis.config.example.json).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Pfad zur CSV-Datei. Überschreibt data_path aus Config/Umgebung.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Name der Freitextspalte. Wenn weggelassen: automatische Erkennung (u. a. CFPB-Spalte).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Anzahl nicht-leerer Texte (Chunk-Reading bis diese Zahl erreicht ist).",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Zeichenkodierung (z. B. utf-8, latin-1) oder 'auto' für automatische Erkennung.",
    )
    parser.add_argument(
        "--csv-sep",
        default=None,
        help=r"Trennzeichen: Komma (Standard), ';', '|', oder \t für Tab.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Zeilenblockgröße beim Einlesen sehr großer CSV.",
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=("de", "en"),
        default=None,
        help="Vorverarbeitungssprache (spaCy-Modell: de_core_news_sm vs en_core_web_sm). Standard: en.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Ausführliche Logausgaben.")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    try:
        settings = merge_analysis_settings(
            config_file_cli=args.config,
            cli_data=args.data,
            cli_text_column=args.text_column,
            cli_sample_size=args.sample_size,
            cli_encoding=args.encoding,
            cli_csv_sep=args.csv_sep,
            cli_chunk_size=args.chunk_size,
            cli_language=args.language,
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error("%s", e)
        return 2

    if settings.config_loaded_from:
        logging.info("Konfiguration geladen: %s", settings.config_loaded_from)

    try:
        path = resolve_data_path(settings.data_path_hint)
        data_raw, load_meta = load_complaints(
            path=path,
            text_column=settings.text_column,
            sample_size=settings.sample_size,
            encoding=settings.encoding,
            csv_separator=settings.csv_separator,
            chunk_size=settings.chunk_size,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        logging.error("%s", e)
        return 1

    os.makedirs("results/visuals", exist_ok=True)

    logging.info("Vorverarbeitungssprache: %s", settings.language)
    data_clean = preprocess_pipeline(data_raw, language=settings.language)

    run_meta = {
        **load_meta,
        "vorverarbeitung_sprache": settings.language,
        "spacy_modell": "de_core_news_sm" if settings.language == "de" else "en_core_web_sm",
    }

    logging.info("Starte Themenextraktion (%s Dokumente nach Vorverarbeitung)…", len(data_clean))
    (tfidf_vec, nmf_mod), (cnt_vec, lda_mod), matrix, score = run_topic_modeling(data_clean)

    logging.info("Qualitätskennzahl (Kohärenz bzw. Fallback): %.4f", score)

    export_topics_to_json(
        nmf_mod,
        tfidf_vec,
        "results/topics_extracted.json",
        n_documents=len(data_clean),
        run_metadata=run_meta,
    )

    fig = plot_top_words(
        nmf_mod,
        tfidf_vec.get_feature_names_out(),
        10,
        "Top Problemfelder der Bürgerbeschwerden",
    )
    fig.savefig("results/visuals/themen_analyse.png", dpi=120)

    logging.info(
        "Fertig — Ergebnisse: results/visuals/themen_analyse.png, results/topics_extracted.json"
    )
    if os.environ.get("MPLBACKEND") != "Agg":
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
