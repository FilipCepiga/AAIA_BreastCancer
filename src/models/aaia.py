import numpy as np
from scipy.spatial.distance import cdist

class AAIAClusterer:
    """
    Artificial Afterimage Algorithm dla problemu klasteryzacji.
    Algorytm oparty na optyce powidoków (Visual Angle, Perceptual Size).
    Używa metryki Manhattan (Cityblock).
    """

    def __init__(self, n_clusters: int = 2, population_size: int = 30,
                 max_iter: int = 200, random_state: int = 42):
        self.k = n_clusters
        self.pop_size = population_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(random_state)

        self.best_centroids_ = None
        self.labels_ = None
        self.sse_history_ = [] 
        self.diversity_hist_ = []
        self.best_sse_ = np.inf

    def _fitness(self, X: np.ndarray, centroids: np.ndarray) -> float:
        dists = cdist(X, centroids, metric='cityblock')
        return float(np.sum(np.min(dists, axis=1)))

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return np.argmin(cdist(X, centroids, metric='cityblock'), axis=1)

    def _init_population(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        population = np.zeros((self.pop_size, self.k, d))
        for i in range(self.pop_size):
            idx = self.rng.choice(n, size=self.k, replace=False)
            population[i] = X[idx].copy()
        return population

    def fit(self, X: np.ndarray) -> 'AAIAClusterer':
        n, d = X.shape
        X_min, X_max = X.min(axis=0), X.max(axis=0)

        print("\n[2/5] Uruchamianie algorytmu Artificial Afterimage Algorithm (AAIA)...")
        print(f"  • Populacja:   {self.pop_size} osobników")
        print(f"  • Iteracje:    {self.max_iter}")
        print(f"  • Klastry:     {self.k}")

        population = self._init_population(X)
        fitness = np.array([self._fitness(X, ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        self.best_sse_ = fitness[best_idx]

        log_interval = max(1, self.max_iter // 10)

        for it in range(self.max_iter):
            worst_idx = np.argmax(fitness)
            worst_individual = population[worst_idx].copy()
            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                pop_ij = population[i]
                
                denom = 2 * pop_ij - best_individual
                denom[denom == 0] = 1e-8

                V = 2 * np.arctan(pop_ij / denom)
                S = V * (pop_ij - best_individual)

                rand_val = self.rng.random(pop_ij.shape)
                term_abs_worst = np.abs(worst_individual - pop_ij)
                term_abs_main = np.abs(S - (best_individual - term_abs_worst))
                
                new_pop_ij = S + (term_abs_main * rand_val)
                new_population[i] = np.clip(new_pop_ij, X_min, X_max)

            new_fitness = np.array([self._fitness(X, ind) for ind in new_population])

            population = new_population.copy()
            fitness = new_fitness.copy()

            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < self.best_sse_:
                self.best_sse_ = fitness[curr_best_idx]
                best_individual = population[curr_best_idx].copy()

            diversity = np.mean(np.std(population.reshape(self.pop_size, -1), axis=0))
            self.sse_history_.append(self.best_sse_)
            self.diversity_hist_.append(diversity)

            if it % log_interval == 0 or it == self.max_iter - 1:
                print(f"  Iter {it+1:4d}/{self.max_iter}  |  "
                      f"Koszt (M-Dist): {self.best_sse_:10.2f}  |  "
                      f"Różnorodność: {diversity:.4f}")

        self.best_centroids_ = best_individual
        self.labels_ = self._assign(X, self.best_centroids_)
        print(f"\n  ✔ AAIA zakończony. Najlepszy koszt: {self.best_sse_:.2f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(X, self.best_centroids_)