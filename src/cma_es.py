from typing import Any, Callable, Dict

import numpy as np
from scipy.stats import qmc


class CMAES:
    """CMA-ES implementation with pluggable sampling strategies."""

    def __init__(self,
                 dimension: int,
                 sigma: float = 0.5,
                 population_size: int = None,
                 sampler_type: str = 'gaussian'):
        """
        Initialize CMA-ES algorithm.

        Args:
            dimension: Problem dimension
            sigma: Initial step size
            population_size: Population size (lambda)
            sampler_type: Type of sampler ('gaussian', 'sobol', 'halton')
        """
        self.n = dimension
        self.sigma = sigma

        # Population size (lambda)
        if population_size is None:
            self.lambda_ = 4 + int(3 * np.log(self.n))
        else:
            self.lambda_ = population_size

        # Selection parameters
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)

        # Adaptation parameters
        self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.c1 = 2 / ((self.n + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs

        # Initialize state
        self.mean = np.zeros(self.n)
        self.C = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)
        self.invsqrtC = np.eye(self.n)

        # Sampling strategy
        self.sampler_type = sampler_type
        self.generation = 0

        # Initialize samplers for low-discrepancy sequences
        if sampler_type == 'sobol':
            self.sobol_sampler = qmc.Sobol(d=self.n, scramble=True)
        elif sampler_type == 'halton':
            self.halton_sampler = qmc.Halton(d=self.n, scramble=True)

        # Statistics
        self.best_fitness = float('inf')
        self.best_solution = None
        self.fitness_history = []

    def _update_eigensystem(self):
        """Update eigendecomposition of covariance matrix."""
        if self.generation % self.n == 0:  # Update every n generations
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # Ensure symmetry
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(self.D, 1e-14))
            self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.T

    def _sample_population(self) -> np.ndarray:
        """Sample new population using specified sampling strategy."""
        if self.sampler_type == 'gaussian':
            return self._sample_gaussian()
        elif self.sampler_type == 'sobol':
            return self._sample_sobol()
        elif self.sampler_type == 'halton':
            return self._sample_halton()
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")

    def _sample_gaussian(self) -> np.ndarray:
        """Standard Gaussian sampling."""
        z = np.random.randn(self.lambda_, self.n)
        return self.mean + self.sigma * (z @ (self.B * self.D).T)

    def _sample_sobol(self) -> np.ndarray:
        """Sobol sequence sampling with Box-Muller transformation."""
        # Generate uniform samples from Sobol sequence
        uniform_samples = self.sobol_sampler.random(self.lambda_)

        # Box-Muller transformation to get Gaussian samples
        z = self._box_muller_transform(uniform_samples)

        # Apply CMA-ES transformation
        return self.mean + self.sigma * (z @ (self.B * self.D).T)

    def _sample_halton(self) -> np.ndarray:
        """Halton sequence sampling with Box-Muller transformation."""
        # Generate uniform samples from Halton sequence
        uniform_samples = self.halton_sampler.random(self.lambda_)

        # Box-Muller transformation to get Gaussian samples
        z = self._box_muller_transform(uniform_samples)

        # Apply CMA-ES transformation
        return self.mean + self.sigma * (z @ (self.B * self.D).T)

    def _box_muller_transform(self, uniform_samples: np.ndarray) -> np.ndarray:
        """Transform uniform samples to Gaussian using Box-Muller method."""
        n_samples, n_dims = uniform_samples.shape

        # Ensure even number of dimensions for Box-Muller pairs
        if n_dims % 2 == 1:
            # Add extra dimension and remove later
            uniform_samples = np.column_stack([uniform_samples, uniform_samples[:, 0]])
            n_dims += 1

        # Reshape for pair processing
        pairs = uniform_samples.reshape(n_samples, n_dims // 2, 2)

        # Box-Muller transformation
        u1, u2 = pairs[:, :, 0], pairs[:, :, 1]

        # Avoid numerical issues
        u1 = np.clip(u1, 1e-10, 1 - 1e-10)
        u2 = np.clip(u2, 1e-10, 1 - 1e-10)

        # Generate Gaussian pairs
        r = np.sqrt(-2 * np.log(u1))
        theta = 2 * np.pi * u2

        z1 = r * np.cos(theta)
        z2 = r * np.sin(theta)

        # Recombine pairs
        z = np.stack([z1, z2], axis=2).reshape(n_samples, n_dims)

        # Remove extra dimension if added
        if self.n % 2 == 1:
            z = z[:, :-1]

        return z

    def optimize(self,
                 objective_func: Callable[[np.ndarray], float],
                 max_evals: int = None,
                 target_fitness: float = 1e-8,
                 verbose: bool = False) -> dict:
        """
        Run CMA-ES optimization.

        Args:
            objective_func: Function to minimize
            max_evals: Maximum number of evaluations
            target_fitness: Target fitness value
            verbose: Print progress

        Returns:
            Dictionary with optimization results
        """
        if max_evals is None:
            max_evals = 10000 * self.n

        evaluations = 0

        while evaluations < max_evals:
            # Sample new population
            population = self._sample_population()

            # Evaluate population
            fitness = np.array([objective_func(x) for x in population])
            evaluations += self.lambda_

            # Sort by fitness
            indices = np.argsort(fitness)
            population = population[indices]
            fitness = fitness[indices]

            # Update best solution
            if fitness[0] < self.best_fitness:
                self.best_fitness = fitness[0]
                self.best_solution = population[0].copy()

            self.fitness_history.append(self.best_fitness)

            # Check convergence
            if self.best_fitness <= target_fitness:
                if verbose:
                    print(f"Converged at generation {self.generation}, evals: {evaluations}")
                break

            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.sum(self.weights[:, np.newaxis] * population[:self.mu], axis=0)

            # Update evolution paths
            mean_diff = self.mean - old_mean
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs)
                                                        * self.mu_eff) * self.invsqrtC @ mean_diff / self.sigma

            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) **
                                                     (2 * evaluations / self.lambda_)) < 1.4 + 2 / (self.n + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc *
                                                               (2 - self.cc) * self.mu_eff) * mean_diff / self.sigma

            # Update covariance matrix
            artmp = (population[:self.mu] - old_mean) / self.sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * \
                (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :] + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            self.C += self.cmu * artmp.T @ np.diag(self.weights) @ artmp

            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.n) - 1))

            # Update eigendecomposition
            self._update_eigensystem()

            self.generation += 1

            if verbose and self.generation % 50 == 0:
                print(f"Generation {self.generation}, Best fitness: {self.best_fitness:.2e}, Sigma: {self.sigma:.2e}")

        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'evaluations': evaluations,
            'generations': self.generation,
            'fitness_history': self.fitness_history,
            'converged': self.best_fitness <= target_fitness
        }
