import pandas as pd

def load_data(filepath):
    # Lädt die Rohdaten aus der CSV-Datei
    df = pd.read_csv(filepath)
    # Extrahiert die Textspalte und entfernt leere Einträge
    # Wir nehmen 5000 Datensätze als repräsentative Stichprobe für die Analyse
    return df['Consumer complaint narrative'].dropna().head(5000)
