# NLP-Themenextraktion aus Bürgerbeschwerden

Dieses Projekt analysiert unstrukturierte Beschwerdetexte, um Kernthemen für kommunale Entscheidungsträger zu identifizieren.

## Installation
1. Repository klonen: `git clone https://github.com/Unadvised7311/NLP-Buergerbeschwerden-Analyse.git`
2. Virtuelle Umgebung erstellen: `python -m venv venv`
3. Abhängigkeiten installieren: `pip install -r requirements.txt`
4. Spacy-Modell laden: `python -m spacy download en_core_web_sm`

## Funktionsweise
- **Preprocessing:** Cleaning, Tokenisierung und Lemmatisierung mit spaCy.
- **Vektorisierung:** Vergleich von TF-IDF und CountVectorizer.
- **Modelle:** Einsatz von NMF (Non-Negative Matrix Factorization) und LDA (Latent Dirichlet Allocation).
- **Validierung:** Qualitätsprüfung mittels Coherence Score.

## Ergebnisse
Die grafischen Auswertungen werden automatisiert im Ordner `results/visuals/` gespeichert.
