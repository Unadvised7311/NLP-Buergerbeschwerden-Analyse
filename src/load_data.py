import pandas as pd

def load_data(filepath):
    # low_memory=False verhindert die Warnung bei großen CSV-Dateien
    df = pd.read_csv(filepath, low_memory=False)

    # Wir prüfen, ob die richtige Spalte existiert
    column_name = 'Consumer complaint narrative'

    if column_name in df.columns:
        # Entferne leere Zeilen und nimm ein Sample für die Performance
        return df[column_name].dropna().head(5000)
    else:
        print(f"❌ Fehler: Spalte '{column_name}' nicht in {filepath} gefunden!")
        return pd.Series()
