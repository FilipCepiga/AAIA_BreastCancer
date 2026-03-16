import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Upewnij się, że folder na wyniki istnieje
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'malignant': '#E74C3C', 'benign': '#2ECC71',
    'cluster0': '#3498DB', 'cluster1': '#E67E22',
    'centroid': '#9B59B6', 'bg': '#F8F9FA', 'grid': '#DEE2E6',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'], 'axes.facecolor': COLORS['bg'],
    'axes.grid': True, 'grid.color': COLORS['grid'],
    'grid.linewidth': 0.7, 'font.family': 'DejaVu Sans', 'font.size': 10,
})

def plot_figure1_overview(df, X_pca2, y_true, pca2):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Rys. 1 – Charakterystyka Zbioru Danych', fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    counts = pd.Series(y_true).map({0: 'Złośliwy (M)', 1: 'Łagodny (B)'}).value_counts()
    bars = ax1.bar(counts.index, counts.values, color=[COLORS['malignant'], COLORS['benign']], width=0.5)
    ax1.set_title('Rozkład Klas', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    for label, name, color in [(0, 'Złośliwy (M)', COLORS['malignant']), (1, 'Łagodny (B)', COLORS['benign'])]:
        mask = y_true == label
        ax2.scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=color, alpha=0.55, s=20, label=name)
    ax2.set_title('PCA 2D – Prawdziwe Etykiety', fontweight='bold')
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    pca_full = PCA(random_state=42).fit(MinMaxScaler().fit_transform(df.drop(columns=['diagnosis', 'diagnosis_label']).values))
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    ax3.plot(range(1, len(cumvar)+1), cumvar, color='#3498DB', marker='o')
    ax3.axhline(95, color='red', linestyle='--', label='95%')
    ax3.set_title('Skumulowana Wariancja PCA', fontweight='bold')
    ax3.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, 'rys1_przeglad_danych.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_figure2_aaia_process(aaia):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Rys. 2 – Zbieżność Optymalizacji AAIA', fontsize=13, fontweight='bold')
    iters = range(1, len(aaia.sse_history_) + 1)

    axes[0].plot(iters, aaia.sse_history_, color='#E74C3C', linewidth=2)
    axes[0].set_title('Zbieżność Funkcji Celu', fontweight='bold')
    
    axes[1].plot(iters, aaia.diversity_hist_, color='#3498DB', linewidth=2)
    axes[1].set_title('Różnorodność Populacji', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rys2_zbieznosc_aaia.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_figure3_clustering_results(X_pca2, aaia_labels, kmeans_labels, y_true, aaia_centroids_pca, pca2, aaia_aligned, kmeans_aligned):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Rys. 3 – Wyniki Klasteryzacji w Przestrzeni PCA 2D', fontsize=13, fontweight='bold')
    
    for ax, title, labels in zip(axes, ['Prawdziwe Etykiety', 'AAIA', 'K-Means'], [y_true, aaia_aligned, kmeans_aligned]):
        for cls, name, color in [(0, 'M', COLORS['malignant']), (1, 'B', COLORS['benign'])]:
            ax.scatter(X_pca2[labels == cls, 0], X_pca2[labels == cls, 1], c=color, alpha=0.55, s=18, label=name)
        if title == 'AAIA':
            ax.scatter(aaia_centroids_pca[:, 0], aaia_centroids_pca[:, 1], c=COLORS['centroid'], s=250, marker='*', edgecolor='black', label='Centroidy AAIA')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        
    plt.savefig(os.path.join(OUTPUT_DIR, 'rys3_wyniki_klasteryzacji.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_figure4_metrics(metrics_list):
    df_m = pd.DataFrame(metrics_list)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Rys. 4 – Porównanie Metryk', fontsize=13, fontweight='bold')
    
    metrics_cols = ['Accuracy', 'ARI', 'NMI', 'Silhouette']
    w = 0.35
    for i, (_, row) in enumerate(df_m.iterrows()):
        vals = [row[m] if m != 'Accuracy' else row[m]/100 for m in metrics_cols]
        axes[0].bar(np.arange(len(metrics_cols)) + i*w, vals, w, label=row['Metoda'], color=['#E74C3C', '#3498DB'][i])
    axes[0].set_xticks(np.arange(len(metrics_cols)) + w/2)
    axes[0].set_xticklabels(metrics_cols)
    axes[0].legend()

    for i, m in enumerate(['Davies-Bouldin', 'Calinski-Harabasz']):
        for j, (_, row) in enumerate(df_m.iterrows()):
            axes[i+1].bar(j, row[m], color=['#E74C3C', '#3498DB'][j], label=row['Metoda'])
        axes[i+1].set_xticks(range(len(df_m)))
        axes[i+1].set_xticklabels(df_m['Metoda'])
        axes[i+1].set_title(m, fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'rys4_metryki.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_figure5_confusion(aaia_aligned, kmeans_aligned, y_true):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, labels, title in zip(axes, [aaia_aligned, kmeans_aligned], ['AAIA', 'K-Means']):
        cm = confusion_matrix(y_true, labels)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=['M', 'B'], yticklabels=['M', 'B'])
        ax.set_title(f'{title} (Acc: {np.diag(cm).sum()/cm.sum()*100:.1f}%)', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, 'rys5_macierze_pomylek.png'), dpi=150, bbox_inches='tight')
    plt.close()