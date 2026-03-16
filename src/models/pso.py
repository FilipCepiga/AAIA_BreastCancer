import numpy as np
from sklearn.cluster import KMeans

class PSOClustering:
    """
    Klasteryzacja przez PSO.
    Każda cząstka reprezentuje zestaw centroidów.
    """
    def __init__(self, n_clusters=2, population_size=50, max_iter=200,
                 w=0.7, c1=1.7, c2=1.7, random_state=42):
        self.n_clusters = n_clusters
        self.pop_size = population_size
        self.max_iter = max_iter
        self.w = w; self.c1 = c1; self.c2 = c2
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
        pos = [best_init.copy() for _ in range(self.pop_size // 3)]
        while len(pos) < self.pop_size:
            pos.append(rng.uniform(lo, hi, shape))
        return pos

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        lo, hi = X.min(0), X.max(0)
        shape = (self.n_clusters, X.shape[1])

        pos = self._init_kmeans_centroids(X, rng)
        vel = [np.zeros(shape) for _ in range(self.pop_size)]
        pbest = [p.copy() for p in pos]
        pbest_fit = [self._fitness(X, p) for p in pbest]
        gbest = pbest[np.argmin(pbest_fit)].copy()

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = rng.rand(*shape), rng.rand(*shape)
                vel[i] = (self.w * vel[i]
                          + self.c1 * r1 * (pbest[i] - pos[i])
                          + self.c2 * r2 * (gbest - pos[i]))
                pos[i] = np.clip(pos[i] + vel[i], lo, hi)
                f = self._fitness(X, pos[i])
                if f < pbest_fit[i]:
                    pbest[i] = pos[i].copy(); pbest_fit[i] = f
                    if f < self._fitness(X, gbest):
                        gbest = pos[i].copy()

        self.best_centroids_ = gbest
        self.labels_ = self._assign(X, gbest)
        return self

    def predict(self, X): return self._assign(X, self.best_centroids_)