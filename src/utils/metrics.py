import numpy as np
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)

def align_labels(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Wyrównuje etykiety klastrów do etykiet rzeczywistych (0/1)."""
    from itertools import permutations
    best_labels, best_acc = labels.copy(), 0.0
    for perm in permutations(range(labels.max() + 1)):
        mapped = np.array([perm[l] for l in labels])
        acc = np.mean(mapped == y_true)
        if acc > best_acc:
            best_acc = acc
            best_labels = mapped
    return best_labels

def compute_metrics(X: np.ndarray, labels: np.ndarray, y_true: np.ndarray, name: str) -> dict:
    """Oblicza kompleksowy zestaw metryk jakości klasteryzacji."""
    aligned = align_labels(labels, y_true)
    ari  = adjusted_rand_score(y_true, labels)
    nmi  = normalized_mutual_info_score(y_true, labels)
    sil  = silhouette_score(X, labels)
    db   = davies_bouldin_score(X, labels)
    ch   = calinski_harabasz_score(X, labels)
    acc  = np.mean(aligned == y_true)

    return {
        'Metoda': name,
        'Accuracy':           round(acc * 100, 2),
        'ARI':                round(ari, 4),
        'NMI':                round(nmi, 4),
        'Silhouette':         round(sil, 4),
        'Davies-Bouldin':     round(db, 4),
        'Calinski-Harabasz':  round(ch, 2),
    }, aligned