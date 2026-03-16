import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def load_and_preprocess():
    """Ładuje Wisconsin Breast Cancer Dataset i przygotowuje dane."""
    print("=" * 65)
    print("  AAIA – Wykrywanie Grup Ryzyka Raka Piersi")
    print("=" * 65)
    print("\n[1/5] Ładowanie danych...")

    data = load_breast_cancer()
    X_raw = data.data
    y_true = data.target
    feature_names = data.feature_names

    df = pd.DataFrame(X_raw, columns=feature_names)
    df['diagnosis'] = y_true
    df['diagnosis_label'] = df['diagnosis'].map({0: 'Złośliwy (M)', 1: 'Łagodny (B)'})

    print(f"  • Próbki ogółem:     {len(df)}")
    print(f"  • Cechy:             {X_raw.shape[1]}")
    print(f"  • Złośliwy (M):      {(y_true == 0).sum()}")
    print(f"  • Łagodny  (B):      {(y_true == 1).sum()}")
    print(f"  • Brak wartości NaN: {df.isnull().sum().sum()}")

    # Normalizacja Min-Max
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # PCA do wizualizacji
    pca2 = PCA(n_components=2, random_state=42)
    X_pca2 = pca2.fit_transform(X_scaled)

    pca10 = PCA(n_components=10, random_state=42)
    X_pca10 = pca10.fit_transform(X_scaled)

    print(f"\n  PCA 2D – wyjaśniona wariancja: "
          f"{pca2.explained_variance_ratio_.sum()*100:.1f}%")

    return df, X_raw, X_scaled, X_pca2, X_pca10, y_true, feature_names, pca2