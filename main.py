from src.load_data import load_data
from src.preprocess import preprocess_pipeline
from src.topic_modeling import run_topic_modeling
from src.visualize import plot_top_words
import os
import matplotlib.pyplot as plt

# Erstellen der Verzeichnisstruktur
os.makedirs('results/visuals', exist_ok=True)

# 1. Daten laden (Consumer Complaint Database)
data_raw = load_data('data/rows.csv')

# 2. Preprocessing Pipeline (Cleaning, Stopwords, Lemmatisierung)
data_clean = preprocess_pipeline(data_raw)

# 3. Topic Modeling (NMF + LDA)
print("Starte Themenextraktion...")
(tfidf_vec, nmf_mod), (cnt_vec, lda_mod), matrix, score = run_topic_modeling(data_clean)

# 4. Validierung (Coherence Score / Feedback-Umsetzung)
print("-" * 30)
print(f"QUALITÄTS-VALIDIERUNG")
print(f"Identifizierter Coherence Score: {score:.4f}")
print("-" * 30)

# 5. Visualisierung für Entscheidungsträger
fig = plot_top_words(nmf_mod, tfidf_vec.get_feature_names_out(), 10, "Top Problemfelder der Bürgerbeschwerden")
fig.savefig('results/visuals/themen_analyse.png')

print("Analyse abgeschlossen. Ergebnisse unter 'results/visuals/' verfügbar.")
plt.show()
