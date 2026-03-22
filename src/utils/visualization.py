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

def plot_figure3_clustering_results(X_pca2, labels_list, names_list, y_true, centroids_list, pca2, aligned_list):
    n_plots = len(labels_list) + 1  # +1 for true labels
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    fig.suptitle('Rys. 3 – Wyniki Klasteryzacji w Przestrzeni PCA 2D', fontsize=13, fontweight='bold')
    
    # True labels
    ax = axes.flat[0]
    for cls, name, color in [(0, 'M', COLORS['malignant']), (1, 'B', COLORS['benign'])]:
        ax.scatter(X_pca2[y_true == cls, 0], X_pca2[y_true == cls, 1], c=color, alpha=0.55, s=18, label=name)
    ax.set_title('Prawdziwe Etykiety', fontweight='bold')
    ax.legend()
    
    for i, (labels, name, aligned, centroids) in enumerate(zip(labels_list, names_list, aligned_list, centroids_list)):
        ax = axes.flat[i+1]
        for cls, cls_name, color in [(0, 'M', COLORS['malignant']), (1, 'B', COLORS['benign'])]:
            ax.scatter(X_pca2[aligned == cls, 0], X_pca2[aligned == cls, 1], c=color, alpha=0.55, s=18, label=cls_name)
        if centroids is not None:
            centroids_pca = pca2.transform(centroids)
            ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c=COLORS['centroid'], s=250, marker='*', edgecolor='black', label=f'Centroidy {name}')
        ax.set_title(name, fontweight='bold')
        ax.legend()
    
    # Hide unused axes
    for j in range(i+2, rows*cols):
        axes.flat[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rys3_wyniki_klasteryzacji.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_figure4_metrics(metrics_list):
    df_m = pd.DataFrame(metrics_list)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Rys. 4 – Porównanie Metryk', fontsize=13, fontweight='bold')
    
    metrics_cols = ['Accuracy', 'ARI', 'NMI', 'Silhouette']
    n_methods = len(df_m)
    w = 0.8 / n_methods  # Adjust width based on number of methods
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#E67E22', '#9B59B6'][:n_methods]
    for i, (_, row) in enumerate(df_m.iterrows()):
        vals = [row[m] if m != 'Accuracy' else row[m]/100 for m in metrics_cols]
        axes[0].bar(np.arange(len(metrics_cols)) + i*w, vals, w, label=row['Metoda'], color=colors[i])
    axes[0].set_xticks(np.arange(len(metrics_cols)) + w*(n_methods-1)/2)
    axes[0].set_xticklabels(metrics_cols)
    axes[0].legend()

    for i, m in enumerate(['Davies-Bouldin', 'Calinski-Harabasz']):
        for j, (_, row) in enumerate(df_m.iterrows()):
            axes[i+1].bar(j, row[m], color=colors[j], label=row['Metoda'] if i==0 else "")
        axes[i+1].set_xticks(range(len(df_m)))
        axes[i+1].set_xticklabels(df_m['Metoda'], rotation=45, ha='right')
        axes[i+1].set_title(m, fontweight='bold')
        if i == 0:
            axes[i+1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rys4_metryki.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_figure5_confusion(aligned_labels_list, method_names, y_true):
    n_methods = len(aligned_labels_list)
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 5))
    fig.suptitle('Rys. 5 – Macierze Pomyłek', fontsize=13, fontweight='bold')
    for ax, labels, title in zip(axes, aligned_labels_list, method_names):
        cm = confusion_matrix(y_true, labels)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=['M', 'B'], yticklabels=['M', 'B'])
        acc = np.diag(cm).sum() / cm.sum() * 100
        ax.set_title(f'{title} (Acc: {acc:.1f}%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rys5_macierze_pomylek.png'), dpi=150, bbox_inches='tight')
    plt.close()