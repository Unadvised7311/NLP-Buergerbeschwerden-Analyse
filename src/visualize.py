import pyLDAvis
import pyLDAvis.lda_model
import matplotlib.pyplot as plt
import numpy as np

def visualize_lda(lda_model, count_matrix, count_vectorizer):
    """Erstellt die interaktive HTML-Visualisierung für LDA."""
    print("Starte LDA-Visualisierung (das kann einen Moment dauern)...")
    try:
        # Wir nutzen den direkten Weg über lda_model, das ist stabiler
        panel = pyLDAvis.lda_model.prepare(
            lda_model,
            count_matrix,
            count_vectorizer,
            mds='tsne'
        )
        pyLDAvis.save_html(panel, 'results/visuals/lda_interaktiv.html')
        print("✅ LDA-Visualisierung gespeichert: results/visuals/lda_interaktiv.html")
    except Exception as e:
        print(f"❌ Fehler bei der LDA-Visualisierung: {e}")

def plot_top_words(model, feature_names, n_top_words, title):
    """Erstellt Balkendiagramme der wichtigsten Wörter pro Thema."""
    n_topics = model.n_components
    # Dynamische Berechnung der Spalten (max 5 nebeneinander)
    n_cols = min(n_topics, 5)
    n_rows = int(np.ceil(n_topics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7, color='skyblue')
        ax.set_title(f"Thema {topic_idx + 1}", fontdict={"fontsize": 14})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=10)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)

    # Leere subplots verstecken
    for i in range(topic_idx + 1, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.4, hspace=0.3)

    # Speichern statt nur Anzeigen (wichtig für dein Projekt-Ordner)
    plt.savefig('results/visuals/nmf_top_words.png', dpi=300, bbox_inches='tight')
    print("✅ NMF-Plot gespeichert: results/visuals/nmf_top_words.png")
    plt.show()
