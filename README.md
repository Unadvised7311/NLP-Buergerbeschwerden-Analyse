# NLP-Themenextraktion aus Beschwerdetexten

Kleines Analyse-Tool: aus Freitextbeschwerden wiederkehrende Themen auslesen (NMF und LDA auf TF-IDF bzw. Häufigkeitsvektoren). Als Rohdaten eignet sich z. B. die Consumer Complaint Database (CFPB), Spalte `Consumer complaint narrative` — analog auch andere CSV mit Beschwerdetext.

## Voraussetzungen

- Python 3.10 bis 3.12  
- virtuelle Umgebung empfohlen  

```bash
python -m venv .venv
source .venv/bin/activate          # bash/zsh; unter Fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

spaCy-Modelle `en_core_web_sm` und `de_core_news_sm` sind über `requirements.txt` eingebunden. Für LDA-Kohärenz optional: `pip install gensim`.

## Datenablage

Die CFPB-`rows.csv` liegt unter **`data/rows.csv`** (lokal beschaffen; zu groß fürs Git). Für Tests ohne Volldatensatz existiert **`data/sample_complaints.csv`**.

## Programmstart

```bash
MPLBACKEND=Agg python main.py
```

Stichprobenumfang über `--sample-size` (Standard: 5000). Bei sehr großen Dateien werden beim Einlesen nur Zeilen mit nicht-leerem Freitext berücksichtigt.

Nützliche Optionen: `--data`, `--text-column`, `--language` (`de` / `en`), `--encoding`, `--csv-sep`, `--chunk-size`, `--config`, `-v`.

Konfiguration alternativ über Umgebungsvariablen (`NLP_DATA_PATH`, `NLP_TEXT_COLUMN`, `NLP_LANGUAGE`, …) oder über eine eigene `analysis.config.json` (Vorlage: `analysis.config.example.json`). Reihenfolge der Übersteuerung: Kommandozeile → Umgebung → JSON → Standard.

Ausgabe nach einem Lauf:

- `results/visuals/themen_analyse.png`  
- `results/topics_extracted.json`  

Demo ohne spaCy (kleines Korpus, nur sklearn/matplotlib):

```bash
python scripts/generate_bundled_results.py
```

## Methodik (überblick)

Vorverarbeitung mit spaCy (Lemma, Stoppwörter, einfache Domänenfilter), dann TF-IDF + NMF sowie CountVectorizer + LDA (`n_topics=5`, `max_df=0.9`, `min_df` abhängig von der Dokumentanzahl). Auswertung und Grafik beziehen sich auf die NMF-Terme.

## Hinweise

Bag-of-Words und ein festes `k` (hier **5** Themen, siehe Methodik) sind bewusste Vereinfachungen — keine automatisch „optimale“ Themenzahl. Beschwerderegister zeigen nur, was gemeldet wird. Für echte Produktivnutzung wären Datenschutz, Zugriffsrechte und stichprobenartige Nachprüfung nötig.
