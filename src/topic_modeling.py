"""NMF + TF-IDF und LDA + CountVectorizer; Kohärenz über gensim wenn installiert."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


def _min_df_for_corpus(n_documents: int) -> int:
    """Passt min_df an kleine Stichproben an (robuster als fixes min_df=5)."""
    if n_documents >= 800:
        return 5
    return max(2, min(5, max(1, n_documents // 150)))


def run_topic_modeling(data):
    docs = list(data)
    n_docs = len(docs)
    min_df = _min_df_for_corpus(n_docs)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=min_df)
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    nmf_model = NMF(n_components=5, random_state=42).fit(tfidf_matrix)

    count_vectorizer = CountVectorizer(max_df=0.9, min_df=min_df)
    count_matrix = count_vectorizer.fit_transform(docs)
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42).fit(count_matrix)

    if HAS_GENSIM:
        tokenized_data = [text.split() for text in docs]
        dictionary = Dictionary(tokenized_data)
        feature_names = count_vectorizer.get_feature_names_out()
        topics = [[feature_names[i] for i in t.argsort()[:-11:-1]] for t in lda_model.components_]

        cm = CoherenceModel(topics=topics, texts=tokenized_data, dictionary=dictionary, coherence="c_v")
        coherence_score = cm.get_coherence()
    else:
        coherence_score = abs(lda_model.score(count_matrix) / 1_000_000)
        print("Hinweis: Gensim nicht installiert — grobe Validierung über LDA.loglikelihood (Baseline).")

    return (tfidf_vectorizer, nmf_model), (count_vectorizer, lda_model), count_matrix, coherence_score


def export_topics_to_json(
    nmf_model,
    tfidf_vectorizer,
    path: str | Path,
    *,
    n_documents: int,
    top_n: int = 10,
    run_metadata: dict | None = None,
) -> None:
    """Schreibt die NMF-Top-Terme pro Thema (primäre Reporting-Basis für Entscheidungsträger)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = tfidf_vectorizer.get_feature_names_out()
    topics_out = []
    for ti, topic in enumerate(nmf_model.components_):
        ind = topic.argsort()[:-top_n - 1 : -1]
        topics_out.append(
            {
                "topic_index": ti + 1,
                "top_terms": [{"term": names[i], "weight": float(topic[i])} for i in ind],
            }
        )

    payload = {
        "n_documents": n_documents,
        "n_topics": int(nmf_model.n_components),
        "top_n_terms": top_n,
        "methodik": (
            "NMF auf TF-IDF nach preprocess_pipeline (spaCy: Lemma, Stoppwörter, Domänenfilter); "
            "Parameter max_df=0.9, min_df adaptiv, n_components=5."
        ),
        "topics": topics_out,
    }
    if run_metadata:
        payload["datenquelle"] = run_metadata
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
