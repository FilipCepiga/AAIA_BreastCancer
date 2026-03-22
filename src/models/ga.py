import numpy as np
from sklearn.cluster import KMeans

class GAClustering:
    """Klasteryzacja przez algorytm genetyczny."""
    def __init__(self, n_clusters=2, population_size=60, max_iter=200,
                 mut_rate=0.2, random_state=42):
        self.n_clusters = n_clusters
        self.pop_size = population_size
        self.max_iter = max_iter
        self.mut_rate = mut_rate
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
        pop = [best_init.copy() for _ in range(self.pop_size // 4)]
        while len(pop) < self.pop_size:
            pop.append(rng.uniform(lo, hi, shape))
        return pop

    def _crossover(self, p1, p2, rng):
        mask = rng.rand(*p1.shape) > 0.5
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, ind, lo, hi, rng):
        mask = rng.rand(*ind.shape) < self.mut_rate
        ind[mask] = rng.uniform(lo, hi, ind.shape)[mask]
        return ind

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        lo, hi = X.min(0), X.max(0)
        shape = (self.n_clusters, X.shape[1])
        pop = self._init_kmeans_centroids(X, rng)

        for _ in range(self.max_iter):
            fits = np.array([self._fitness(X, p) for p in pop])
            inv  = 1.0 / (fits + 1e-9)
            probs = inv / inv.sum()
            new_pop = []
            for _ in range(self.pop_size):
                i1, i2 = rng.choice(self.pop_size, 2, replace=False, p=probs)
                child = self._crossover(pop[i1].copy(), pop[i2], rng)
                child = self._mutate(child, lo, hi, rng)
                new_pop.append(child)
            pop = new_pop

        fits = [self._fitness(X, p) for p in pop]
        best = pop[np.argmin(fits)]
        self.best_centroids_ = best
        self.labels_ = self._assign(X, best)
        return self

    def predict(self, X): return self._assign(X, self.best_centroids_)