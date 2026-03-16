import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans

# Importy z naszych lokalnych modułów
from src.data_loader import load_and_preprocess
from src.models.aaia import AAIAClusterer
from src.models.firefly import FireflyClustering
from src.models.pso import PSOClustering
from src.models.ga import GAClustering
from src.utils.metrics import compute_metrics
from src.utils.visualization import (
    plot_figure1_overview,
    plot_figure2_aaia_process,
    plot_figure3_clustering_results,
    plot_figure4_metrics,
    plot_figure5_confusion
)

warnings.filterwarnings('ignore')

def main():
    # Ustawienie globalnego seeda dla powtarzalności w mainie
    np.random.seed(42)

    # 1. Pobieranie danych
    df, X_raw, X_scaled, X_pca2, X_pca10, y_true, feature_names, pca2 = load_and_preprocess()

    # 2. Uruchomienie modelu AAIA
    aaia = AAIAClusterer(
        n_clusters=2,
        population_size=30,
        max_iter=150,
        random_state=42,
    )
    aaia.fit(X_scaled)

    firefly = FireflyClustering(
        n_clusters=2,
        population_size=30,
        max_iter=150,
        random_state=42,
    )
    firefly.fit(X_scaled)

    pso = PSOClustering(
        n_clusters=2,
        population_size=30,
        max_iter=150,
        random_state=42,
    )
    pso.fit(X_scaled)

    ga = GAClustering(
        n_clusters=2,
        population_size=30,
        max_iter=150,
        random_state=42,
    )
    ga.fit(X_scaled)

    # 3. K-Means jako punkt odniesienia (Baseline)
    print("\n[3/5] K-Means (punkt odniesienia)...")
    km = KMeans(n_clusters=2, n_init=20, max_iter=500, random_state=42)
    km.fit(X_scaled)
    kmeans_labels = km.labels_

    # 4. Obliczanie metryk
    print("\n[4/5] Obliczanie metryk...")
    aaia_metrics, aaia_aligned = compute_metrics(X_scaled, aaia.labels_, y_true, 'AAIA')
    firefly_metrics, firefly_aligned = compute_metrics(X_scaled, firefly.labels_, y_true, 'Firefly')
    pso_metrics, pso_aligned = compute_metrics(X_scaled, pso.labels_, y_true, 'PSO')
    ga_metrics, ga_aligned = compute_metrics(X_scaled, ga.labels_, y_true, 'GA')
    kmeans_metrics, kmeans_aligned = compute_metrics(X_scaled, kmeans_labels, y_true, 'K-Means')
    metrics_list = [aaia_metrics, firefly_metrics, pso_metrics, ga_metrics, kmeans_metrics]

    print("\n" + "=" * 65)
    print("  WYNIKI METRYK JAKOŚCI")
    print("=" * 65)
    df_results = pd.DataFrame(metrics_list).set_index('Metoda')
    print(df_results.to_string())

    # 5. Rysowanie wykresów
    print("\n[5/5] Generowanie wizualizacji...")
    
    plot_figure1_overview(df, X_pca2, y_true, pca2)
    plot_figure2_aaia_process(aaia)
    plot_figure3_clustering_results(
        X_pca2, 
        [aaia.labels_, firefly.labels_, pso.labels_, ga.labels_, kmeans_labels], 
        ['AAIA', 'Firefly', 'PSO', 'GA', 'K-Means'], 
        y_true, 
        [aaia.best_centroids_, firefly.best_centroids_, pso.best_centroids_, ga.best_centroids_, None], 
        pca2, 
        [aaia_aligned, firefly_aligned, pso_aligned, ga_aligned, kmeans_aligned]
    )
    plot_figure4_metrics(metrics_list)
    plot_figure5_confusion([aaia_aligned, firefly_aligned, pso_aligned, ga_aligned, kmeans_aligned], ['AAIA', 'Firefly', 'PSO', 'GA', 'K-Means'], y_true)

    print("\n" + "=" * 65)
    print("  ZAKOŃCZONO POMYŚLNIE")
    print("  Pliki graficzne zapisano w folderze: outputs/")
    print("=" * 65)

if __name__ == '__main__':
    main()