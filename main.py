from src.load_data import load_data
from src.preprocess import preprocess_pipeline
from src.topic_modeling import run_topic_modeling
from src.visualize import visualize_lda, plot_top_words
import joblib
import os
import matplotlib.pyplot as plt
os.makedirs('results/visuals', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
# Automatische Erstellung der benötigten Ordner
folders = ['results/models', 'results/visuals', 'data']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"✅ Ordner erstellt: {folder}")

# Daten laden und vorverarbeiten
data_raw = load_data('data/rows.csv')
data_clean = preprocess_pipeline(data_raw)

# Topic Modeling durchführen
(tfidf_vectorizer, nmf_model), (count_vectorizer, lda_model), count_matrix = run_topic_modeling(data_clean)

# Visualisierungen erstellen
visualize_lda(lda_model, count_matrix, count_vectorizer)
plot_top_words(nmf_model, tfidf_vectorizer.get_feature_names_out(), 10, "Top Begriffe pro NMF-Thema (TF-IDF)")

# Modelle speichern
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/visuals', exist_ok=True)

joblib.dump(lda_model, 'results/models/lda_final.pkl')
joblib.dump(count_vectorizer, 'results/models/count_vec.pkl')
joblib.dump(nmf_model, 'results/models/nmf_final.pkl')
joblib.dump(tfidf_vectorizer, 'results/models/tfidf_vec.pkl')

# Visualisierungen speichern
plt.figure(figsize=(20, 10))
plot_top_words(nmf_model, tfidf_vectorizer.get_feature_names_out(), 10, "Beschwerde-Themen (NMF)")
plt.savefig('results/visuals/nmf_themen_uebersicht.png', dpi=300)
