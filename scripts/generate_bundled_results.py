#!/usr/bin/env python3
"""Erzeugt Demo-PNG/JSON mit sklearn (ohne spaCy)."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import NMF  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample_complaints.csv"
OUT_JSON = ROOT / "results" / "topics_extracted.json"
OUT_PNG = ROOT / "results" / "visuals" / "themen_analyse.png"


def simple_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(w for w in s.split() if len(w) > 2)


def main() -> int:
    if not SAMPLE_CSV.is_file():
        print(f"Fehlt: {SAMPLE_CSV}", file=sys.stderr)
        return 1

    rows = []
    with SAMPLE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(simple_clean(row["Consumer complaint narrative"]))

    n_docs = len(rows)
    min_df = 5 if n_docs >= 800 else max(2, min(5, max(1, n_docs // 150)))

    vec = TfidfVectorizer(max_df=0.9, min_df=min_df)
    X = vec.fit_transform(rows)
    nmf = NMF(n_components=5, random_state=42).fit(X)
    names = vec.get_feature_names_out()

    topics_out = []
    top_n = 10
    for ti, topic in enumerate(nmf.components_):
        ind = topic.argsort()[:-top_n - 1 : -1]
        topics_out.append(
            {
                "topic_index": ti + 1,
                "top_terms": [{"term": names[i], "weight": float(topic[i])} for i in ind],
            }
        )

    payload = {
        "n_documents": n_docs,
        "n_topics": 5,
        "top_n_terms": top_n,
        "methodik": (
            "NMF auf TF-IDF (gebündelte Demo ohne spaCy; dieselben Kernparameter wie in "
            "src.topic_modeling nach Lemma-Stoppwort-Filterung in preprocess_pipeline)."
        ),
        "vorverarbeitung_demo": (
            "Nur Kleinbuchstaben, Sonderzeichen entfernt, kurze Tokens verworfen — "
            "Ersetzt spaCy hier zum Nachbau der Artefakte."
        ),
        "topics": topics_out,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(20, 10), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[:-top_n - 1 : -1]
        top_features = [names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, color="teal")
        ax.set_title(f"Thema {topic_idx + 1}")
        ax.invert_yaxis()
        for s in "top right left".split():
            ax.spines[s].set_visible(False)
    fig.suptitle("Top Problemfelder der Bürgerbeschwerden (gebündeltes Demo-Ergebnis)", fontsize=18)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"Geschrieben: {OUT_JSON}\nGeschrieben: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
