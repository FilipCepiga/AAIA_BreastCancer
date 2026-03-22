import numpy as np
from sklearn.cluster import KMeans

class FireflyClustering:
    """Klasteryzacja przez algorytm świetlika."""
    def __init__(self, n_clusters=2, population_size=50, max_iter=200,
                 alpha=1.0, beta0=2.0, gamma=0.5, random_state=42):
        self.n_clusters = n_clusters
        self.pop_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha; self.beta0 = beta0; self.gamma = gamma
        self.random_state = random_state
        self.best_centroids_ = None; self.labels_ = None

    def _dist(self, X, c): return np.sum(np.abs(X - c), axis=1)

    def _assign(self, X, cents):
        return np.argmin(np.column_stack([self._dist(X, c) for c in cents]), axis=1)

    def _fitness(self, X, cents):
        lbl = self._assign(X, cents)
        return sum(
            self._dist(X[lbl == k], cents[k]).sum()
            for k in range(self.n_clusters) if (lbl == k).any()
        )

    def _init_kmeans_centroids(self, X, rng):
        """Inicjalizacja populacji za pomocą K-means++"""
        km = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=5, random_state=self.random_state)
        km.fit(X)
        best_init = km.cluster_centers_.copy()
        
        shape = (self.n_clusters, X.shape[1])
        lo, hi = X.min(0), X.max(0)
        pop = [best_init.copy() for _ in range(self.pop_size // 3)]
        while len(pop) < self.pop_size:
            pop.append(rng.uniform(lo, hi, shape))
        return pop

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        lo, hi = X.min(0), X.max(0)
        shape = (self.n_clusters, X.shape[1])
        pop = self._init_kmeans_centroids(X, rng)
        fits = [self._fitness(X, p) for p in pop]

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fits[j] < fits[i]:
                        r2 = np.sum((pop[i] - pop[j]) ** 2)
                        beta = self.beta0 * np.exp(-self.gamma * r2)
                        pop[i] = (pop[i]
                                  + beta * (pop[j] - pop[i])
                                  + self.alpha * (rng.rand(*shape) - 0.5))
                        pop[i] = np.clip(pop[i], lo, hi)
                        fits[i] = self._fitness(X, pop[i])

        best = pop[np.argmin(fits)]
        self.best_centroids_ = best
        self.labels_ = self._assign(X, best)
        return self

    def predict(self, X): return self._assign(X, self.best_centroids_)